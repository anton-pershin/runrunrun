FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Install Python, pip, and basic tooling
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Provide the python alias expected by many scripts
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Ensure pip installs the correct CUDA-enabled torch backend for vLLM
ENV VLLM_INSTALL_TORCH_BACKEND=auto

# Install vLLM nightly build (required for Qwen3.5 support)
#    --extra-index-url https://wheels.vllm.ai/nightly \
# cc0d565f40814b7406ac7a420725c54ce6ebd116
RUN pip install --no-cache-dir --break-system-packages \
    --extra-index-url https://wheels.vllm.ai/cc0d565f40814b7406ac7a420725c54ce6ebd116 https://wheels.vllm.ai/cc0d565f40814b7406ac7a420725c54ce6ebd116/vllm-0.16.1rc1.dev143%2Bgcc0d565f4-cp38-abi3-manylinux_2_31_x86_64.whl

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Example entrypoint for manual serving (uncomment to use)
# CMD ["vllm", "serve", "Qwen/Qwen3.5-9B", \
#      "--reasoning-parser", "qwen3", \
#      "--tensor-parallel-size", "1", \
#      "--max-model-len", "262144"]
