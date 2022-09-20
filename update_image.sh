#!/bin/bash
docker build -t gcr.io/context-ml/lightningmnist:ai_platform -f ai_platform.dockerfile ./
docker push gcr.io/context-ml/lightningmnist:ai_platform

docker build -t gcr.io/context-ml/lightningmnist:vertex_ai -f vertex_ai.dockerfile ./
docker push gcr.io/context-ml/lightningmnist:vertex_ai
