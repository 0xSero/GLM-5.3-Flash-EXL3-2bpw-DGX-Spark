FROM ghcr.io/0xsero/glm53-flash-exl3-k2-dflash@sha256:6be6de479a5c8c6b8ce9ce42a7be3a2f79e9eeed854ddd4d73e3fc407da88a4d
COPY kv_cache_utils.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
RUN python3 -m py_compile /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py
LABEL org.opencontainers.image.description="Experimental DFlash manager512 capacity candidate; runtime acceptance pending"
