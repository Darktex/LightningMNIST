# LightningMNIST
This is just a simple example on using PyTorch Lightning to orchestrate multi-box distributed training.

I have put both AI Platform and Vertex AI dockerfiles and launch scripts.

To launch, look at the `scripts` directory.

# Launching on Cloud AI Platform

Launch a distributed job on V100s on AI platform like this:

`scripts/ai_platform_submit.sh -g V100 -d true`

Launch a one-box multi-gpu (so still distributed job) on V100s like this:

`scripts/ai_platform_submit.sh -g V100`

# Launching on Vertex AI

The equivalent scripts for Vertex AI are these:

`scripts/vertex_ai_submit.sh -g V100 -d true`