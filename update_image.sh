#!/bin/bash
docker build -t gcr.io/context-ml/lightningmnist:caip ./
docker push gcr.io/context-ml/lightningmnist:caip