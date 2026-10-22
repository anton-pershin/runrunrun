FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Install vLLM and dependencies (using --break-system-packages for Docker container)
RUN pip install --no-cache-dir --break-system-packages vllm>=0.11.0

# Expose the default vLLM port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the vLLM server
#CMD ["vllm", "serve", "ai21labs/AI21-Jamba-Reasoning-3B", \
#     "--mamba-ssm-cache-dtype", "float32", \
#     "--reasoning-parser", "deepseek_r1", \
#     "--enable-auto-tool-choice", \
#     "--tool-call-parser", "hermes"]
