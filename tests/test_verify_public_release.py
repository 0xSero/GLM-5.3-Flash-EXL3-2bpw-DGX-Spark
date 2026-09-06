import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import verify_public_release as verifier


class PublicReleaseVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        (self.root / "evidence").mkdir()

        self.model_revision = "a" * 40
        self.github_revision = "b" * 40
        self.weight = b"weight"
        self.manifest = json.dumps(
            {
                "files": [
                    {
                        "path": "weight.safetensors",
                        "bytes": len(self.weight),
                        "sha256": hashlib.sha256(self.weight).hexdigest(),
                    }
                ]
            },
            sort_keys=True,
        ).encode()
        self.index = b"index"
        self.bitmap = b"bitmap"
        self.structural = b"structural"
        self.docker_index = "sha256:" + "c" * 64
        self.docker_arm64 = "sha256:" + "d" * 64
        self.github_url = "https://github.com/example/release"

        release = {
            "artifact": {
                "manifest_sha256": hashlib.sha256(self.manifest).hexdigest(),
                "weight_files": 1,
                "weight_bytes": len(self.weight),
            },
            "release": {
                "model_repository": "example/model",
                "model_revision": self.model_revision,
                "docker_image": "ghcr.io/example/image:tag",
                "docker_digest": self.docker_index,
                "docker_arm64_manifest_digest": self.docker_arm64,
                "github_repository": self.github_url,
                "github_revision": self.github_revision,
            },
        }
        structure = {
            "model_index": {"sha256": hashlib.sha256(self.index).hexdigest()},
            "tier_bitmap_sha256": hashlib.sha256(self.bitmap).hexdigest(),
            "structural_evidence_manifest": {
                "sha256": hashlib.sha256(self.structural).hexdigest()
            },
        }
        (self.root / "release.json").write_text(json.dumps(release))
        (self.root / "evidence/artifact-structure.json").write_text(
            json.dumps(structure)
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def request_json(self, url: str, headers=None):
        del headers
        if "/api/models/" in url:
            return (
                {
                    "private": False,
                    "gated": False,
                    "disabled": False,
                    "sha": self.model_revision,
                    "siblings": [
                        {
                            "rfilename": "weight.safetensors",
                            "size": len(self.weight),
                            "lfs": {
                                "sha256": hashlib.sha256(self.weight).hexdigest()
                            },
                        }
                    ],
                },
                {},
            )
        if "api.github.com" in url:
            return ({"sha": self.github_revision}, {})
        raise AssertionError(url)

    def download(self, url: str) -> bytes:
        if url.endswith("EXL3_MANIFEST.json"):
            return self.manifest
        if url.endswith("model.safetensors.index.json"):
            return self.index
        if url.endswith("tier_bitmap.json"):
            return self.bitmap
        if url.endswith("STRUCTURAL_EVIDENCE_MANIFEST.json"):
            return self.structural
        if url.endswith("README.md"):
            return f"{self.github_url}\n{self.docker_index}\n".encode()
        raise AssertionError(url)

    def registry_manifest(self, repository: str, digest: str, token: str):
        self.assertEqual(repository, "example/image")
        self.assertEqual(token, "anonymous-token")
        if digest == self.docker_index:
            return (
                {
                    "manifests": [
                        {
                            "digest": self.docker_arm64,
                            "platform": {"os": "linux", "architecture": "arm64"},
                        }
                    ]
                },
                digest,
            )
        self.assertEqual(digest, self.docker_arm64)
        return ({"schemaVersion": 2}, digest)

    def test_complete_anonymous_release_passes(self) -> None:
        with (
            mock.patch.object(verifier, "request_json", self.request_json),
            mock.patch.object(verifier, "download", self.download),
            mock.patch.object(
                verifier, "anonymous_registry_token", return_value="anonymous-token"
            ),
            mock.patch.object(verifier, "registry_manifest", self.registry_manifest),
        ):
            result = verifier.verify(self.root)
        self.assertTrue(result["pass"], result["errors"])
        self.assertEqual(result["huggingface_weight_files"], 1)

    def test_private_hub_and_registry_failure_fail_closed(self) -> None:
        def private_hub(url: str, headers=None):
            payload, response_headers = self.request_json(url, headers)
            if "/api/models/" in url:
                payload["private"] = True
            return payload, response_headers

        with (
            mock.patch.object(verifier, "request_json", private_hub),
            mock.patch.object(verifier, "download", self.download),
            mock.patch.object(
                verifier,
                "anonymous_registry_token",
                side_effect=RuntimeError("private"),
            ),
        ):
            result = verifier.verify(self.root)
        self.assertFalse(result["pass"])
        self.assertIn("Hugging Face repository is not public", result["errors"])
        self.assertIn(
            "GHCR anonymous verification failed: RuntimeError", result["errors"]
        )


if __name__ == "__main__":
    unittest.main()
