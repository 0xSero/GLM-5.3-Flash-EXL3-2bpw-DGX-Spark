#!/usr/bin/env python3
"""Verify the immutable release from anonymous public endpoints only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


OCI_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def request_json(url: str, headers: dict[str, str] | None = None) -> tuple[dict, object]:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response), response.headers


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def anonymous_registry_token(repository: str) -> str:
    url = f"https://ghcr.io/v2/{repository}/manifests/latest"
    try:
        urllib.request.urlopen(
            urllib.request.Request(url, headers={"Accept": OCI_ACCEPT}), timeout=30
        )
    except urllib.error.HTTPError as error:
        challenge = error.headers.get("WWW-Authenticate", "")
    else:
        raise RuntimeError("GHCR did not issue an anonymous Bearer challenge")

    fields = dict(re.findall(r'(\w+)="([^"]+)"', challenge))
    if not fields.get("realm") or not fields.get("service"):
        raise RuntimeError("GHCR Bearer challenge is incomplete")
    query = urllib.parse.urlencode(
        {
            "service": fields["service"],
            "scope": f"repository:{repository}:pull",
        }
    )
    token_payload, _ = request_json(f"{fields['realm']}?{query}")
    token = token_payload.get("token") or token_payload.get("access_token")
    if not token:
        raise RuntimeError("GHCR did not grant an anonymous pull token")
    return str(token)


def registry_manifest(repository: str, digest: str, token: str) -> tuple[dict, str]:
    url = f"https://ghcr.io/v2/{repository}/manifests/{digest}"
    payload, headers = request_json(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": OCI_ACCEPT},
    )
    observed = headers.get("Docker-Content-Digest", "")
    if observed != digest:
        raise RuntimeError(f"registry digest mismatch for {digest}")
    return payload, observed


def registry_blob(repository: str, digest: str, token: str) -> bytes:
    url = f"https://ghcr.io/v2/{repository}/blobs/{digest}"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def registry_blob_exists(repository: str, digest: str, token: str) -> None:
    url = f"https://ghcr.io/v2/{repository}/blobs/{digest}"
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"}, method="HEAD"
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"registry blob HEAD failed for {digest}")


def verify(root: Path) -> dict:
    release = json.loads((root / "release.json").read_text())
    structure = json.loads((root / "evidence/artifact-structure.json").read_text())
    model = release["release"]
    artifact = release["artifact"]
    errors: list[str] = []
    docker_index = str(model.get("docker_digest") or "")
    docker_arm64 = str(model.get("docker_arm64_manifest_digest") or "")
    dflash = release.get("dflash_release") or {}
    dflash_digest = str(dflash.get("docker_digest") or "")
    dflash_platform = None
    dflash_layer_count = 0

    revision = str(model.get("model_revision") or "")
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        errors.append("model_revision is not an immutable commit")

    hub_info: dict = {}
    if revision:
        try:
            repo_id = model["model_repository"]
            encoded_repo = urllib.parse.quote(repo_id, safe="/")
            hub_info, _ = request_json(
                f"https://huggingface.co/api/models/{encoded_repo}?blobs=true"
            )
            if hub_info.get("private") is not False:
                errors.append("Hugging Face repository is not public")
            if hub_info.get("gated") not in (False, None):
                errors.append("Hugging Face repository is gated")
            if hub_info.get("disabled") not in (False, None):
                errors.append("Hugging Face repository is disabled")
            if hub_info.get("sha") != revision:
                errors.append("Hugging Face head differs from model_revision")

            siblings = hub_info.get("siblings") or []
            weights = [
                item
                for item in siblings
                if str(item.get("rfilename", "")).endswith(".safetensors")
            ]
            if len(weights) != artifact["weight_files"]:
                errors.append("Hugging Face weight-file count differs")
            if sum(int(item.get("size") or 0) for item in weights) != artifact["weight_bytes"]:
                errors.append("Hugging Face weight-byte total differs")
            if any(not (item.get("lfs") or {}).get("sha256") for item in weights):
                errors.append("Hugging Face weight LFS digest metadata is incomplete")

            manifest_url = (
                f"https://huggingface.co/{encoded_repo}/resolve/{revision}/"
                "EXL3_MANIFEST.json"
            )
            manifest_bytes = download(manifest_url)
            if hashlib.sha256(manifest_bytes).hexdigest() != artifact["manifest_sha256"]:
                errors.append("Hugging Face structural manifest digest differs")
            manifest = json.loads(manifest_bytes)
            expected_weights = {
                item["path"]: {"size": int(item["bytes"]), "sha256": item["sha256"]}
                for item in manifest.get("files", [])
            }
            observed_weights = {
                item["rfilename"]: {
                    "size": int(item.get("size") or 0),
                    "sha256": (item.get("lfs") or {}).get("sha256"),
                }
                for item in weights
            }
            if observed_weights != expected_weights:
                errors.append("Hugging Face weight closure or per-file digest differs")

            immutable_root = f"https://huggingface.co/{encoded_repo}/resolve/{revision}"
            for path, expected_sha256 in (
                ("model.safetensors.index.json", structure["model_index"]["sha256"]),
                ("tier_bitmap.json", structure["tier_bitmap_sha256"]),
                (
                    "STRUCTURAL_EVIDENCE_MANIFEST.json",
                    structure["structural_evidence_manifest"]["sha256"],
                ),
            ):
                if hashlib.sha256(download(f"{immutable_root}/{path}")).hexdigest() != expected_sha256:
                    errors.append(f"Hugging Face {path} digest differs")

            card = download(f"{immutable_root}/README.md").decode(errors="replace")
            if model["github_repository"] not in card:
                errors.append("Hugging Face model card lacks the public runtime repository")
            if docker_index not in card:
                errors.append("Hugging Face model card lacks the immutable Docker digest")
        except Exception as error:  # fail closed and avoid response-body disclosure
            errors.append(f"Hugging Face anonymous verification failed: {type(error).__name__}")

    image = str(model.get("docker_image") or "")
    repository = image.removeprefix("ghcr.io/").split("@", 1)[0].split(":", 1)[0]
    try:
        token = anonymous_registry_token(repository)
        index, _ = registry_manifest(repository, docker_index, token)
        arm64 = [
            item
            for item in index.get("manifests", [])
            if item.get("platform", {}).get("os") == "linux"
            and item.get("platform", {}).get("architecture") == "arm64"
        ]
        if len(arm64) != 1 or arm64[0].get("digest") != docker_arm64:
            errors.append("GHCR ARM64 manifest digest differs")
        else:
            registry_manifest(repository, docker_arm64, token)
    except Exception as error:  # fail closed and avoid token/response disclosure
        errors.append(f"GHCR anonymous verification failed: {type(error).__name__}")

    if dflash.get("publication_status", "").startswith("public_"):
        dflash_image = str(dflash.get("docker_image") or "")
        dflash_repository = (
            dflash_image.removeprefix("ghcr.io/").split("@", 1)[0].split(":", 1)[0]
        )
        try:
            token = anonymous_registry_token(dflash_repository)
            manifest, _ = registry_manifest(
                dflash_repository, dflash_digest, token
            )
            config_digest = str((manifest.get("config") or {}).get("digest") or "")
            if config_digest != dflash.get("docker_config_digest"):
                errors.append("DFlash GHCR config digest differs")
            config = json.loads(registry_blob(dflash_repository, config_digest, token))
            dflash_platform = f"{config.get('os')}/{config.get('architecture')}"
            if dflash_platform != "linux/arm64":
                errors.append("DFlash GHCR platform is not linux/arm64")
            layers = manifest.get("layers") or []
            dflash_layer_count = len(layers)
            for item in layers:
                registry_blob_exists(dflash_repository, str(item["digest"]), token)
        except Exception as error:
            errors.append(
                f"DFlash GHCR anonymous verification failed: {type(error).__name__}"
            )

    github_revision = str(model.get("github_revision") or "")
    try:
        github_repo = model["github_repository"].removeprefix("https://github.com/")
        commit, _ = request_json(
            f"https://api.github.com/repos/{github_repo}/commits/{github_revision}"
        )
        if commit.get("sha") != github_revision:
            errors.append("GitHub revision is not publicly resolvable")
    except Exception as error:
        errors.append(f"GitHub anonymous verification failed: {type(error).__name__}")

    return {
        "schema": "glm53-public-release-verification-v1",
        "pass": not errors,
        "model_repository": model.get("model_repository"),
        "model_revision": revision or None,
        "huggingface_weight_files": len(
            [
                item
                for item in hub_info.get("siblings", [])
                if str(item.get("rfilename", "")).endswith(".safetensors")
            ]
        ),
        "docker_index_digest": docker_index,
        "docker_arm64_manifest_digest": docker_arm64,
        "dflash_manifest_digest": dflash_digest or None,
        "dflash_platform": dflash_platform,
        "dflash_layer_count": dflash_layer_count,
        "github_revision": github_revision,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path(__file__).parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.root.resolve())
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
