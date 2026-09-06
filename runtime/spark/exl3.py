# SPDX-License-Identifier: Apache-2.0
"""EXL3/MCG trellis quantization for GLM-5.3-Flash routed experts.

Derived from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at commit
32db610d9207a42e2688a6994d3bfaf7af96eecb, with fail-closed support for
campaign artifacts whose TP4 slices are already selected per Spark rank.

Checkpoint ABI (brandonmusic/GLM-5.3-Flash-tr3-4bpw):
  quant_method=exl3, codebook=mcg, scope=glm53_routed_experts_only
  per expert matrix: trellis (int16) + suh/svh (fp16) + mcg (int32 marker)

Non-routed tensors stay native (UnquantizedLinearMethod). Experts never
expand to a persistent BF16 weight; LinearEXL3 / exllamav3_ext runs the
trellis GEMM. Normal TP shards gate/up column-wise and down row-wise. The
rank-stacked TP1 path retains a TP4 archive as four virtual rank slices per
physical expert and sums their contributions in the fused MoE accumulator.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.utils import set_weight_attrs

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)

EXLLAMAV3_COMMIT = "c5d9c657966ffeeaa9353f0cc899f18629da4a13"
EXLLAMAV3_VERSION = "0.0.43"
MCG_MULTIPLIER = 0xCBAC1FED
MCG_MARKER_SIGNED_INT32 = -877912083
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")
RANK_STACKED_TP = 4
RANK_STACKED_INTERMEDIATE = 512
_RANK_IN_NAME = re.compile(r"(?:^|[._])rank(?P<rank>\d+)(?=[._])")
SWIGLU_LIMIT_DEFAULT = 10.0
TEMP_ROWS_FUSED = 128
MOE_ACT_SILU = 0
# Shared fused scratch: decode is sequential across layers.
_FUSED_TEMP_CACHE: dict[
    tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def _narrow_tp(
    tensor: torch.Tensor, dim: int, tp_rank: int, tp_size: int
) -> torch.Tensor:
    if tp_size <= 1:
        return tensor
    size = int(tensor.shape[dim])
    if size % tp_size:
        raise ValueError(
            f"EXL3 TP shard: dim {dim} size {size} is not divisible by tp={tp_size}"
        )
    chunk = size // tp_size
    return tensor.narrow(dim, chunk * tp_rank, chunk).contiguous()


def shard_exl3_col(
    loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Gate/up: trellis dim 1 and svh dim 0 are column-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 1, tp_rank, tp_size)
    if suffix == "svh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def shard_exl3_row(
    loaded: torch.Tensor, suffix: str, tp_rank: int, tp_size: int
) -> torch.Tensor:
    """Down: trellis dim 0 and suh dim 0 are row-parallel."""
    if suffix == "trellis":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    if suffix == "suh":
        return _narrow_tp(loaded, 0, tp_rank, tp_size)
    return loaded.contiguous()


def _install_exllamav3_namespace() -> None:
    """Load LinearEXL3 without running exllamav3/__init__.py (FlashAttention)."""
    if "exllamav3.modules.quant.exl3" in sys.modules:
        return
    import exllamav3_ext  # noqa: F401  — compiled extension must exist

    spec = importlib.util.find_spec("exllamav3")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("exllamav3 package is not installed in this image")
    package_root = Path(list(spec.submodule_search_locations)[0])

    # Stub only packages whose __init__.py pulls FlashAttention / serving extras.
    # Leave .ext, .util, and .modules.quant as real modules so LinearEXL3 loads.
    for name, path in (
        ("exllamav3", package_root),
        ("exllamav3.modules", package_root / "modules"),
        ("exllamav3.model", package_root / "model"),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__file__ = str(path / "__init__.py")
        module.__package__ = name
        module.__path__ = [str(path)]
        sys.modules[name] = module

    if "exllamav3.model.config" not in sys.modules:
        config = types.ModuleType("exllamav3.model.config")
        config.__file__ = str(package_root / "model/config.py")
        config.__package__ = "exllamav3.model"
        config.Config = type("Config", (), {})
        sys.modules[config.__name__] = config


def load_linear_exl3_cls():
    _install_exllamav3_namespace()
    return importlib.import_module("exllamav3.modules.quant.exl3").LinearEXL3


def make_linear_exl3(
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float16,
):
    """Build a LinearEXL3 over already-sharded packed tensors. No BF16 expand."""
    cls = load_linear_exl3_cls()
    return cls(
        config=None,
        in_features=int(suh.numel()),
        out_features=int(svh.numel()),
        trellis=trellis.contiguous(),
        suh=suh.contiguous(),
        svh=svh.contiguous(),
        mcg=mcg.contiguous(),
        out_dtype=out_dtype,
        transformers_fix=True,
    )


def execute_exl3_linear(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    mcg: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Real EXL3 expert GEMM entry (LinearEXL3 / exllamav3_ext)."""
    inner = make_linear_exl3(trellis, suh, svh, mcg, out_dtype=torch.float16)
    return inner.forward(x.contiguous().half(), {}, out_dtype=out_dtype)


def fused_moe_enabled() -> bool:
    return os.environ.get("EXL3_FUSED_MOE", "1") != "0"


def load_exllamav3_ext():
    import exllamav3_ext

    return exllamav3_ext


def _exl3_moe_accepts_num_active(fn) -> bool:
    try:
        import inspect

        if "num_active" in inspect.signature(fn).parameters:
            return True
    except (TypeError, ValueError):
        pass
    doc = getattr(fn, "__doc__", None) or ""
    return "num_active" in doc or "arg29" in doc or doc.count("arg") >= 30


def pin_exl3_expert_map(
    layer: torch.nn.Module, device: torch.device
) -> torch.Tensor | None:
    """Move expert_map onto `device` once. CUDA graph capture forbids a CPU→GPU copy."""
    emap = getattr(layer, "expert_map", None)
    if emap is None:
        return None
    if emap.device != device or emap.dtype != torch.long:
        layer.expert_map = emap.to(device=device, dtype=torch.long)
    return layer.expert_map


def map_topk_to_local(
    ids: torch.Tensor,
    n_local: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    """ids (T, K) global expert ids → local ids, invalid/non-local → n_local sentinel.

    `expert_map` must already live on `ids.device` (see pin_exl3_expert_map).
    """
    flat = ids.reshape(-1)
    if expert_map is None:
        invalid = (flat < 0) | (flat >= n_local)
        return torch.where(invalid, flat.new_full(flat.shape, n_local), flat)
    if expert_map.device != flat.device or expert_map.dtype != torch.long:
        raise RuntimeError(
            "EXL3 expert_map is not pinned to the hidden-state device; "
            "call pin_exl3_expert_map before fused apply (CUDA graphs forbid the copy)"
        )
    n_global = int(expert_map.numel())
    safe = flat.clamp(min=0, max=max(n_global - 1, 0))
    mapped = expert_map[safe] if n_global else flat.new_full(flat.shape, n_local)
    invalid = (flat < 0) | (flat >= n_global) | (mapped < 0) | (mapped >= n_local)
    return torch.where(invalid, flat.new_full(flat.shape, n_local), mapped)


def expand_rank_stacked_routing(
    ids: torch.Tensor,
    weights: torch.Tensor,
    n_physical: int,
    rank_stacked_tp: int,
    expert_map: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand physical expert routes into rank-slice virtual expert routes.

    Each virtual expert computes one tensor-parallel slice of the same physical
    expert. Repeating the router weight makes the fused MoE accumulation equal
    the original TP all-reduce: ``weight * sum(rank_slice_output)``.
    """
    if rank_stacked_tp != RANK_STACKED_TP:
        raise RuntimeError(
            "EXL3 rank-stacked TP1 only supports four archive ranks; "
            f"got rank_stacked_tp={rank_stacked_tp}"
        )
    if ids.shape != weights.shape:
        raise ValueError(
            f"EXL3 routing ids/weights shape mismatch: {tuple(ids.shape)} != "
            f"{tuple(weights.shape)}"
        )
    local = map_topk_to_local(ids, n_physical, expert_map).reshape(ids.shape)
    ranks = torch.arange(rank_stacked_tp, dtype=local.dtype, device=local.device)
    expanded_ids = local.unsqueeze(-1) * rank_stacked_tp + ranks
    invalid = local.unsqueeze(-1) >= n_physical
    expanded_ids = torch.where(
        invalid,
        expanded_ids.new_full(expanded_ids.shape, n_physical * rank_stacked_tp),
        expanded_ids,
    )
    expanded_weights = weights.unsqueeze(-1).expand(*weights.shape, rank_stacked_tp)
    return (
        expanded_ids.reshape(*ids.shape[:-1], ids.shape[-1] * rank_stacked_tp),
        expanded_weights.reshape(
            *weights.shape[:-1], weights.shape[-1] * rank_stacked_tp
        ),
    )


def apply_exl3_python_loop(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
    *,
    only_experts: set[int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unique-expert LinearEXL3 loop. `only_experts` is local ids (fat-expert fallback)."""
    tokens, hidden = x2d.shape
    if out is None:
        out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    unique = torch.unique(ids)
    for raw in unique.tolist():
        e_raw = int(raw)
        if e_raw < 0:
            continue
        e = e_raw
        if expert_map is not None:
            mapped = int(expert_map[e].item()) if expert_map.numel() > e else e
            if mapped < 0:
                continue
            e = mapped
        if e >= len(inners):
            continue
        if only_experts is not None and e not in only_experts:
            continue
        token_idx, k_pos = (ids == int(raw)).nonzero(as_tuple=True)
        h = x2d.index_select(0, token_idx)
        pack = inners[e]
        gate = pack["gate"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        up = pack["up"].forward(h.contiguous().half(), {}, out_dtype=torch.float32)
        act = F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)
        down = pack["down"].forward(
            act.contiguous().half(), {}, out_dtype=torch.float32
        )
        scale = weights[token_idx, k_pos].unsqueeze(-1).to(dtype=torch.float32)
        out.index_add_(0, token_idx, down * scale)
    return out


def build_exl3_fused_state(
    layer: torch.nn.Module, inners: list[dict[str, Any]]
) -> None:
    """Pointer tables + fused temps, once after load. No per-token alloc."""
    import exllamav3_ext

    device = next(iter(inners[0].values())).trellis.device
    n_exp = len(inners)
    hidden = int(layer._exl3_hidden_size)
    intermediate = int(layer._exl3_intermediate_local)

    def _ptrs(which: str, attr: str) -> torch.Tensor:
        return torch.tensor(
            [int(getattr(pack[which], attr).data_ptr()) for pack in inners],
            dtype=torch.int64,
            device=device,
        )

    layer._exl3_ptrs = {
        "gate_trellis": _ptrs("gate", "trellis"),
        "gate_suh": _ptrs("gate", "suh"),
        "gate_svh": _ptrs("gate", "svh"),
        "up_trellis": _ptrs("up", "trellis"),
        "up_suh": _ptrs("up", "suh"),
        "up_svh": _ptrs("up", "svh"),
        "down_trellis": _ptrs("down", "trellis"),
        "down_suh": _ptrs("down", "suh"),
        "down_svh": _ptrs("down", "svh"),
    }
    idx = int(device.index) if device.index is not None else 0
    concurrency = int(exllamav3_ext.exl3_moe_max_concurrency(idx))
    if concurrency < 1:
        concurrency = 1
    key = (str(device), hidden, intermediate, concurrency)
    temps = _FUSED_TEMP_CACHE.get(key)
    if temps is None:
        temps = (
            torch.empty(
                (concurrency, TEMP_ROWS_FUSED, hidden),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, TEMP_ROWS_FUSED, hidden),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, TEMP_ROWS_FUSED, intermediate),
                dtype=torch.float16,
                device=device,
            ),
            torch.empty(
                (concurrency, TEMP_ROWS_FUSED, intermediate),
                dtype=torch.float16,
                device=device,
            ),
        )
        _FUSED_TEMP_CACHE[key] = temps
    layer._exl3_fused_temps = temps
    layer._exl3_fused_concurrency = concurrency
    layer._exl3_k = int(layer._exl3_bits)


def apply_exl3_fused_moe(
    x2d: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    layer: torch.nn.Module,
    inners: list[dict[str, Any]],
    expert_map: torch.Tensor | None,
    limit: float,
) -> torch.Tensor:
    """One exl3_moe launch per layer. Experts with count > 128 fall back to LinearEXL3."""
    import exllamav3_ext

    tokens, hidden = x2d.shape
    n_exp = len(inners)
    ptrs = getattr(layer, "_exl3_ptrs", None)
    temps = getattr(layer, "_exl3_fused_temps", None)
    if not ptrs or temps is None:
        raise RuntimeError("EXL3 fused pointer tables were not built after weight load")

    local = map_topk_to_local(ids, n_exp, expert_map)
    topk = int(ids.shape[-1])
    flat_token = torch.arange(
        tokens, device=x2d.device, dtype=torch.long
    ).repeat_interleave(topk)
    flat_weight = weights.reshape(-1).to(dtype=torch.float16)
    order = local.argsort()
    token_sorted = flat_token[order]
    weight_sorted = flat_weight[order]
    # scatter_add stays on GPU. torch.bincount can host-stage and break CUDA graphs.
    expert_count = torch.zeros(n_exp + 1, dtype=torch.long, device=local.device)
    expert_count.scatter_add_(
        0, local.long(), torch.ones(local.shape, dtype=torch.long, device=local.device)
    )
    out = torch.zeros(tokens, hidden, dtype=torch.float32, device=x2d.device)
    xh = x2d.contiguous().half()

    counts = expert_count[:n_exp]
    fn = exllamav3_ext.exl3_moe
    # -1 = unknown active count: max-concurrency grid, no .item() host sync.
    n_active_host = -1 if _exl3_moe_accepts_num_active(fn) else None

    k = int(getattr(layer, "_exl3_k", 4))
    args = (
        xh,
        out,
        expert_count,
        token_sorted,
        weight_sorted,
        temps[0],
        temps[1],
        temps[2],
        temps[3],
        MOE_ACT_SILU,
        k,
        k,
        k,
        ptrs["gate_trellis"],
        ptrs["gate_suh"],
        ptrs["gate_svh"],
        ptrs["up_trellis"],
        ptrs["up_suh"],
        ptrs["up_svh"],
        ptrs["down_trellis"],
        ptrs["down_suh"],
        ptrs["down_svh"],
        True,
        False,
        True,
        False,
        True,
        False,
        float(limit),
    )
    if n_active_host is not None:
        fn(*args, n_active_host)
    else:
        fn(*args)

    if tokens > TEMP_ROWS_FUSED:
        fat = (counts > TEMP_ROWS_FUSED).nonzero(as_tuple=False).view(-1)
        if fat.numel():
            apply_exl3_python_loop(
                x2d,
                ids,
                weights,
                inners,
                expert_map,
                limit,
                only_experts=set(int(i) for i in fat.tolist()),
                out=out,
            )
    return out


def apply_exl3_experts(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    layer: torch.nn.Module,
    *,
    limit: float = SWIGLU_LIMIT_DEFAULT,
    fused: bool | None = None,
) -> torch.Tensor:
    """Shipped routed-expert apply. `fused=None` honors EXL3_FUSED_MOE."""
    inners = getattr(layer, "_exl3_inners", None)
    if not inners:
        raise RuntimeError("EXL3 experts were not built after weight load")
    tokens, hidden = x.shape[-2], x.shape[-1]
    x2d = x.reshape(tokens, hidden)
    ids = topk_ids.reshape(tokens, -1).to(torch.long)
    weights = topk_weights.reshape(tokens, -1)
    expert_map = pin_exl3_expert_map(layer, x2d.device)
    rank_stacked_tp = int(getattr(layer, "_exl3_rank_stacked_tp", 1))
    if rank_stacked_tp > 1:
        n_physical = int(getattr(layer, "_exl3_physical_experts", 0))
        if n_physical <= 0 or len(inners) != n_physical * rank_stacked_tp:
            raise RuntimeError(
                "EXL3 rank-stacked expert allocation is inconsistent with routing"
            )
        ids, weights = expand_rank_stacked_routing(
            ids, weights, n_physical, rank_stacked_tp, expert_map
        )
        expert_map = None
    have_ptrs = bool(getattr(layer, "_exl3_ptrs", None))
    if fused is True and not have_ptrs:
        raise RuntimeError("EXL3 fused apply requested but pointer tables are missing")
    use_fused = (fused_moe_enabled() if fused is None else bool(fused)) and have_ptrs
    if use_fused:
        try:
            import exllamav3_ext

            use_fused = hasattr(exllamav3_ext, "exl3_moe")
        except Exception:
            use_fused = False
    if use_fused:
        out = apply_exl3_fused_moe(x2d, ids, weights, layer, inners, expert_map, limit)
        layer._exl3_last_apply = "fused"
    else:
        out = apply_exl3_python_loop(x2d, ids, weights, inners, expert_map, limit)
        layer._exl3_last_apply = "loop"
    return out.to(dtype=x.dtype)


def _suffix_from_mapped_name(weight_name: str) -> str:
    tail = weight_name.rsplit(".", 1)[-1]
    for suffix in EXL3_SUFFIXES:
        if tail == suffix or tail.endswith("_" + suffix):
            return suffix
    raise ValueError(f"not an EXL3 packed name: {weight_name}")


def _rank_from_mapped_name(weight_name: str) -> int | None:
    match = _RANK_IN_NAME.search(weight_name)
    return int(match.group("rank")) if match else None


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):
    """Routed-experts-only EXL3/MCG. Dense / shared / attention stay native."""

    def __init__(
        self,
        bits: int = 4,
        codebook: str = "mcg",
        scope: str = "glm53_routed_experts_only",
        rank_stacked_tp: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.bits = int(bits)
        self.codebook = str(codebook)
        self.scope = str(scope)
        self.raw_config = dict(kwargs)
        self.rank_stacked_tp = None if rank_stacked_tp is None else int(rank_stacked_tp)
        if self.rank_stacked_tp is not None:
            self.raw_config["rank_stacked_tp"] = self.rank_stacked_tp
        if self.codebook != "mcg":
            raise ValueError(
                f"this overlay only implements codebook=mcg; got {self.codebook!r}"
            )
        if self.bits not in (2, 3, 4, 5, 6):
            raise ValueError(f"unsupported EXL3 bits={self.bits}")
        if self.rank_stacked_tp not in (None, RANK_STACKED_TP):
            raise ValueError(
                "this runtime only implements rank_stacked_tp=4; "
                f"got {self.rank_stacked_tp!r}"
            )
        if self.rank_stacked_tp and self.raw_config.get("pre_sliced", False):
            raise ValueError("rank_stacked_tp and pre_sliced are mutually exclusive")

    def get_name(self) -> str:
        return "exl3"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        # LinearEXL3 uses CUDA >= Ampere; GB10 is SM121.
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        skip = {
            "bits",
            "codebook",
            "scope",
            "rank_stacked_tp",
            "quant_method",
            # tr3 ships a 37 MiB per-tensor ledger; keep it off the config object.
            "tensor_storage",
        }
        return cls(
            bits=int(config.get("bits", 4)),
            codebook=str(config.get("codebook", "mcg")),
            scope=str(config.get("scope", "glm53_routed_experts_only")),
            rank_stacked_tp=config.get("rank_stacked_tp"),
            **{k: v for k, v in config.items() if k not in skip},
        )

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> str | None:
        method = str((hf_quant_cfg or {}).get("quant_method", "")).lower()
        if method == "exl3":
            return "exl3"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

        if isinstance(layer, RoutedExperts):
            return Exl3MoEMethod(layer.moe_config, self)
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        return None


class Exl3MoEMethod(FusedMoEMethodBase):
    """Packed MCG trellis experts: create/load packed tensors, LinearEXL3 apply."""

    def __init__(self, moe, quant_config: Exl3Config) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.bits = quant_config.bits
        self._logged = False

    def get_fused_moe_quant_config(
        self, layer: "RoutedExperts"
    ) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(
        self,
        layer: "RoutedExperts",
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del params_dtype
        from vllm.distributed import get_tensor_model_parallel_world_size

        rank_stacked_tp = self.quant_config.rank_stacked_tp or 1
        live_tp = get_tensor_model_parallel_world_size()
        if rank_stacked_tp > 1:
            if live_tp != 1:
                raise RuntimeError(
                    "EXL3 rank-stacked archive requires live tensor parallel size 1; "
                    f"got tp={live_tp}"
                )
            if intermediate_size_per_partition % rank_stacked_tp:
                raise ValueError(
                    "EXL3 rank-stacked intermediate size is not divisible by its "
                    f"archive TP: {intermediate_size_per_partition} / {rank_stacked_tp}"
                )
            intermediate_local = intermediate_size_per_partition // rank_stacked_tp
            if intermediate_local != RANK_STACKED_INTERMEDIATE:
                raise ValueError(
                    "EXL3 GLM-5.3 rank-stacked ABI requires a 512-wide slice; "
                    f"got {intermediate_local}"
                )
        else:
            intermediate_local = intermediate_size_per_partition

        if hidden_size % 16 or intermediate_local % 16:
            raise ValueError(
                "EXL3 trellis tiles are 16-wide; "
                f"hidden={hidden_size} intermediate_local={intermediate_local}"
            )
        k_words = self.bits * 16
        in_tiles = hidden_size // 16
        out_tiles = intermediate_local // 16

        extra = {k: v for k, v in extra_weight_attrs.items() if k != "weight_loader"}

        def new_packed() -> dict[str, Parameter]:
            return {
                "w13_trellis": Parameter(
                    torch.empty(
                        num_experts, 2, in_tiles, out_tiles, k_words, dtype=torch.int16
                    ),
                    requires_grad=False,
                ),
                "w13_suh": Parameter(
                    torch.empty(num_experts, 2, hidden_size, dtype=torch.float16),
                    requires_grad=False,
                ),
                "w13_svh": Parameter(
                    torch.empty(
                        num_experts, 2, intermediate_local, dtype=torch.float16
                    ),
                    requires_grad=False,
                ),
                "w13_mcg": Parameter(
                    torch.empty(num_experts, 2, 1, dtype=torch.int32),
                    requires_grad=False,
                ),
                "w2_trellis": Parameter(
                    torch.empty(
                        num_experts, out_tiles, in_tiles, k_words, dtype=torch.int16
                    ),
                    requires_grad=False,
                ),
                "w2_suh": Parameter(
                    torch.empty(num_experts, intermediate_local, dtype=torch.float16),
                    requires_grad=False,
                ),
                "w2_svh": Parameter(
                    torch.empty(num_experts, hidden_size, dtype=torch.float16),
                    requires_grad=False,
                ),
                "w2_mcg": Parameter(
                    torch.empty(num_experts, 1, dtype=torch.int32),
                    requires_grad=False,
                ),
            }

        def bind(param: Parameter, archive_rank: int | None) -> None:
            set_weight_attrs(param, extra)
            param.weight_loader = self._load_exl3
            param._exl3_owner = layer
            param._exl3_archive_rank = archive_rank

        if rank_stacked_tp > 1:
            # Expert mapping changes ``gate_proj.rank0.trellis`` into
            # ``w13_rank0.trellis``. Real submodules preserve that name all the
            # way to our loader without patching the model architecture.
            for archive_rank in range(rank_stacked_tp):
                w13 = torch.nn.Module()
                w2 = torch.nn.Module()
                packed = new_packed()
                for suffix in EXL3_SUFFIXES:
                    p13 = packed[f"w13_{suffix}"]
                    p2 = packed[f"w2_{suffix}"]
                    w13.register_parameter(suffix, p13)
                    w2.register_parameter(suffix, p2)
                    bind(p13, archive_rank)
                    bind(p2, archive_rank)
                layer.add_module(f"w13_rank{archive_rank}", w13)
                layer.add_module(f"w2_rank{archive_rank}", w2)
        else:
            # w13_* is stacked [expert, {gate=0, up=1}, ...] so the stock
            # expert_params_mapping (experts.w13_ + suffix) hits these names.
            for name, param in new_packed().items():
                layer.register_parameter(name, param)
                bind(param, None)
        if hasattr(layer, "w13_weight") or hasattr(layer, "w2_weight"):
            raise RuntimeError(
                "EXL3 create_weights must not allocate dense expert weights"
            )

        layer._exl3_hidden_size = hidden_size
        layer._exl3_intermediate_local = intermediate_local
        layer._exl3_k_words = k_words
        layer._exl3_bits = self.bits
        layer._exl3_rank_stacked_tp = rank_stacked_tp
        layer._exl3_physical_experts = num_experts

    def _load_exl3(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str = "w1",
        expert_id: int = 0,
        return_success: bool = False,
    ) -> bool | None:
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        layer = param
        # param is the Parameter; expert_id is already physical. Map to local
        # via the owning module if present on the weight_loader closure... we
        # look up from param's __dict__ after register. RoutedExperts.weight_loader
        # maps global→local; glm5next calls *our* loader, so map here.
        owner = getattr(param, "_exl3_owner", None)
        if owner is not None:
            local_id = owner._map_global_expert_id_to_local_expert_id(expert_id)
            if local_id == -1:
                return False if return_success else None
            expert_id = local_id

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        suffix = _suffix_from_mapped_name(weight_name)
        archive_rank = _rank_from_mapped_name(weight_name)
        loaded = loaded_weight.detach().contiguous()
        # Safetensors preserves the campaign's scalar MCG marker as shape (),
        # while RoutedExperts stores one marker per projection as shape (1,).
        # They are the same single int32 value; normalize only that exact ABI.
        if suffix == "mcg" and loaded.ndim == 0:
            loaded = loaded.reshape(1)

        pre_sliced = bool(self.quant_config.raw_config.get("pre_sliced", False))
        rank_stacked_tp = self.quant_config.rank_stacked_tp
        if rank_stacked_tp is not None:
            if tp_size != 1 or tp_rank != 0:
                raise RuntimeError(
                    "EXL3 rank-stacked archive requires the live TP1 rank 0 process; "
                    f"got rank={tp_rank}/{tp_size}"
                )
            if archive_rank is None or not 0 <= archive_rank < rank_stacked_tp:
                raise RuntimeError(
                    "EXL3 rank-stacked tensor name must retain rank0..rank3: "
                    f"{weight_name}"
                )
            expected_rank = getattr(param, "_exl3_archive_rank", None)
            if expected_rank != archive_rank:
                raise RuntimeError(
                    "EXL3 rank-stacked tensor was mapped to the wrong packed parameter: "
                    f"name rank={archive_rank}, destination rank={expected_rank}"
                )
        elif archive_rank is not None:
            raise RuntimeError(
                "EXL3 checkpoint contains rank-sliced names but rank_stacked_tp is not "
                f"declared: {weight_name}"
            )
        if pre_sliced:
            declared_tp = int(self.quant_config.raw_config.get("pre_sliced_tp", -1))
            declared_rank = int(self.quant_config.raw_config.get("pre_sliced_rank", -1))
            if declared_tp != tp_size or declared_rank != tp_rank:
                raise RuntimeError(
                    "EXL3 pre-sliced rank view does not match the live TP process: "
                    f"view rank={declared_rank}/{declared_tp}, "
                    f"runtime rank={tp_rank}/{tp_size}"
                )

        if shard_id in ("w1", "w3"):
            shard_idx = 0 if shard_id == "w1" else 1
            sharded = (
                loaded
                if pre_sliced or rank_stacked_tp is not None
                else shard_exl3_col(loaded, suffix, tp_rank, tp_size)
            )
            dest = param.data[expert_id, shard_idx]
        elif shard_id == "w2":
            sharded = (
                loaded
                if pre_sliced or rank_stacked_tp is not None
                else shard_exl3_row(loaded, suffix, tp_rank, tp_size)
            )
            dest = param.data[expert_id]
        else:
            raise ValueError(f"unknown EXL3 shard_id={shard_id}")

        if tuple(dest.shape) != tuple(sharded.shape):
            raise RuntimeError(
                f"EXL3 load shape mismatch {weight_name} shard={shard_id} "
                f"expert={expert_id}: dest {tuple(dest.shape)} != "
                f"loaded {tuple(sharded.shape)}"
            )
        dest.copy_(sharded)
        return True if return_success else None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        rank_stacked_tp = int(getattr(layer, "_exl3_rank_stacked_tp", 1))
        if not hasattr(layer, "w13_trellis") and not hasattr(layer, "w13_rank0"):
            return
        # Bind owner for any late loads; stitch LinearEXL3 handles.
        packed_ranks: list[tuple[Any, Any]] = []
        if rank_stacked_tp > 1:
            for archive_rank in range(rank_stacked_tp):
                w13 = getattr(layer, f"w13_rank{archive_rank}")
                w2 = getattr(layer, f"w2_rank{archive_rank}")
                packed_ranks.append((w13, w2))
                for module in (w13, w2):
                    for param in module.parameters():
                        param._exl3_owner = layer
        else:
            for name in (
                "w13_trellis",
                "w13_suh",
                "w13_svh",
                "w13_mcg",
                "w2_trellis",
                "w2_suh",
                "w2_svh",
                "w2_mcg",
            ):
                getattr(layer, name)._exl3_owner = layer
            packed_ranks.append(
                (
                    types.SimpleNamespace(
                        trellis=layer.w13_trellis,
                        suh=layer.w13_suh,
                        svh=layer.w13_svh,
                        mcg=layer.w13_mcg,
                    ),
                    types.SimpleNamespace(
                        trellis=layer.w2_trellis,
                        suh=layer.w2_suh,
                        svh=layer.w2_svh,
                        mcg=layer.w2_mcg,
                    ),
                )
            )

        mcg13 = torch.cat([w13.mcg.reshape(-1) for w13, _ in packed_ranks])
        mcg2 = torch.cat([w2.mcg.reshape(-1) for _, w2 in packed_ranks])
        if not torch.all(mcg13 == MCG_MARKER_SIGNED_INT32) or not torch.all(
            mcg2 == MCG_MARKER_SIGNED_INT32
        ):
            raise RuntimeError(
                "EXL3 mcg marker is not the MCG int32 0xCBAC1FED / "
                f"{MCG_MARKER_SIGNED_INT32}; packed ABI mismatch"
            )

        n_physical = int(getattr(layer, "_exl3_physical_experts", 0))
        if n_physical <= 0:
            n_physical = int(packed_ranks[0][0].trellis.shape[0])
        inners: list[dict[str, Any]] = []
        # Expert-major ordering matches expand_rank_stacked_routing:
        # virtual_id = physical_expert * rank_stacked_tp + archive_rank.
        for e in range(n_physical):
            for w13, w2 in packed_ranks:
                gate = make_linear_exl3(
                    w13.trellis[e, 0], w13.suh[e, 0], w13.svh[e, 0], w13.mcg[e, 0]
                )
                up = make_linear_exl3(
                    w13.trellis[e, 1], w13.suh[e, 1], w13.svh[e, 1], w13.mcg[e, 1]
                )
                down = make_linear_exl3(w2.trellis[e], w2.suh[e], w2.svh[e], w2.mcg[e])
                inners.append({"gate": gate, "up": up, "down": down})
        n_exp = len(inners)
        layer._exl3_inners = inners
        fused_ok = False
        fused_err = None
        if fused_moe_enabled():
            try:
                import exllamav3_ext

                if hasattr(exllamav3_ext, "exl3_moe"):
                    build_exl3_fused_state(layer, inners)
                    fused_ok = True
                else:
                    fused_err = "exllamav3_ext.exl3_moe missing"
            except Exception as exc:
                fused_err = repr(exc)
                layer._exl3_ptrs = None
        if not self._logged:
            if fused_ok:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "rank_stacked_tp=%s "
                    "fused_moe=exl3_moe concurrency=%s "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    rank_stacked_tp,
                    getattr(layer, "_exl3_fused_concurrency", "?"),
                )
            else:
                logger.info(
                    "EXL3 MCG trellis engaged for routed experts: bits=%s "
                    "experts_local=%s hidden=%s intermediate_local=%s "
                    "rank_stacked_tp=%s "
                    "fused_moe=python_loop (%s) "
                    "(no BF16 expert reconstruct at load)",
                    self.bits,
                    n_exp,
                    layer._exl3_hidden_size,
                    layer._exl3_intermediate_local,
                    rank_stacked_tp,
                    fused_err or "EXL3_FUSED_MOE=0",
                )
            self._logged = True

    def apply(
        self,
        layer: "RoutedExperts",
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        limit = getattr(self.moe, "swiglu_limit", None) or SWIGLU_LIMIT_DEFAULT
        return apply_exl3_experts(x, topk_ids, topk_weights, layer, limit=float(limit))
