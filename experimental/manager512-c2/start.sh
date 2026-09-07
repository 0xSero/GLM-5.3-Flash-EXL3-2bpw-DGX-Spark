#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${GLM53_MODEL_ROOT:?set GLM53_MODEL_ROOT to the sealed exact-2.0 artifact}"
IMAGE="${GLM53_IMAGE:?set GLM53_IMAGE to the manager512 image or build tag}"
DRAFT_ROOT="${GLM53_DRAFT_ROOT:?set GLM53_DRAFT_ROOT to the separately downloaded pinned DFlash2 draft}"
CACHE_ROOT="${GLM53_CACHE_ROOT:-$PWD/.cache/vllm-glm53-exact2-tp1}"
SERVED_MODEL_NAME="${GLM53_SERVED_MODEL_NAME:-glm-5.3-flash-exl3-k2-single-spark}"
HOST="${GLM53_HOST:-127.0.0.1}"
PORT="${GLM53_PORT:-18080}"

python3 - "$MODEL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
assembly = json.loads((root / "assembly-status.json").read_text())
config = json.loads((root / "config.json").read_text())["quantization_config"]
index = json.loads((root / "model.safetensors.index.json").read_text())
references = set(index["weight_map"].values())
root_shards = {path.name for path in root.glob("*.safetensors")}

assert assembly["state"] == "COMPLETE"
assert assembly["structural_pass"] is True
assert config["quant_method"] == "exl3"
assert config["bits"] == 2
assert config["rank_stacked_tp"] == 4
assert config["runtime_tensor_parallel_size"] == 1
assert len(index["weight_map"]) == 583090
assert len(references) == 133
assert references == root_shards
assert all("/" not in name and (root / name).is_file() for name in references)
PY

if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  printf 'foreign CUDA compute process present; refusing runtime launch\n' >&2
  exit 1
fi
mkdir -p "$CACHE_ROOT"

exec docker run -d --restart unless-stopped \
  --name glm53-flash-manager512-c2-20260906 \
  --gpus all \
  --ipc host \
  --network host \
  -v "$MODEL_ROOT":/model:ro \
  -v "$DRAFT_ROOT":/dflash:ro \
  -v "$CACHE_ROOT":/root/.cache/vllm \
  -e HF_HUB_OFFLINE=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SAFETENSORS_DROP_PAGE_CACHE=1 \
  -e SAFETENSORS_LOAD_DEVICE=cuda:0 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_EXL3_TRELLIS_MIN_M=1 \
  -e VLLM_EXL3_TRELLIS_MAX_M=32 \
  -e VLLM_EXL3_PREFILL_TRELLIS=1 \
  -e EXL3_FUSED_MOE=1 \
  "$IMAGE" /model \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --no-enable-expert-parallel \
  --quantization exl3 \
  --load-format safetensors \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --block-size 64 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 204800 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 2048 \
  --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0.1 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["all"],"cudagraph_capture_sizes":[8,16],"max_cudagraph_capture_size":16}' \
  --generation-config vllm \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"/dflash","num_speculative_tokens":7,"kv_cache_dtype":"fp8_e4m3"}'
