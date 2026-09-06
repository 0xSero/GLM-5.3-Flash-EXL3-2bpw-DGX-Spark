# Third-party notices

- Z.AI: GLM-5.3-Flash-BF16 source model, MIT. Source revision is recorded in release.json.
- TurboDerp / ExLlamaV3: EXL3 format, encoder, and runtime extension. The
  container build pins ExLlamaV3 revision
  `c5d9c657966ffeeaa9353f0cc899f18629da4a13`; ExLlamaV3 is MIT licensed.
- Dione: independent selective-conversion workflow lineage.
- Brandon M. Music: earlier MIT GLM-5.2 EXL3/TR3 lineage and reproducibility methods. Permitted reference artifact: brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw at f79c9167690ca705e877ae4dc55a841d1aae1247. Do not imply use of later GLM-5.3 materials.
- Inco: the optional [GLM-5.3-Flash-DFlash2 draft](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a), revision `bf582e4eacc1810f76656d1811693ff6c6737d2a`, is separately licensed under CC-BY-NC-ND-4.0. It is an external dependency of the acceleration experiment, not part of the MIT model weights or a bundled Docker image. The accelerated configuration has not passed release acceptance.

No private calibration payload, ShapleyMCG source, weights, corpus, or generated
artifact is included in this repository.

The runtime overlay uses the MiaAI GLM Flash EXL3 image pinned to
`sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`
as its base. The installed vLLM distribution reports
`0.1.dev20051+g487ecf187` and Apache-2.0; the package version resolves to
upstream vLLM commit `487ecf187d3dfe74d2cf6119a92881dba403c219`.
