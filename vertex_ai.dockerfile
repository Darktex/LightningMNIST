FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-11:latest

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Copies the trainer code to the docker image.
COPY . .
RUN pip install -e .

# ENV NCCL_SOCKET_IFNAME=eth0
# ENV NCCL_IB_DISABLE=1
ENV NCCL_DEBUG=INFO

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "lightningmnist.train"]
