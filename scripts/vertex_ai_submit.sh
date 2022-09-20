#!/bin/bash
## Let's do some admin work to find out the variables to be used here
SCRIPT=$(basename ${BASH_SOURCE[0]})
SCRIPTS_DIR=$(dirname $0)
BOLD='\e[1;31m' # Bold Red
REV='\e[1;32m'  # Bold Green

#Help function
function HELP {
  echo -e "${REV}Basic usage:${OFF} ${BOLD}$SCRIPT -s SOURCE_DIR -t TARGET_DIR ${OFF}"\\n
  echo -e "${REV}The following switches are recognized. $OFF "
  echo -e "${REV}-g ${OFF}  --Sets the GPU model to either T4, V100 or A100. ${OFF} Default is ${BOLD} T4 ${OFF}"
  echo -e "${REV}-h ${OFF}  --Displays this help message. No further functions are performed."\\n
  echo -e "Example: ${BOLD}$SCRIPT -g V100${OFF}"\\n
  exit 1
}

# GPU_MODEL='T4'
DISTRIBUTED='false'
REGION=us-central1
MULTIBOX_PREFIX=''
NUM_NODES=1

# In case you wanted to check what variables were passed
echo "flags = $*"

while getopts :d:g:r:h FLAG; do
  case $FLAG in
  d)
    DISTRIBUTED=$OPTARG
    MULTIBOX_PREFIX='multibox_'
    NUM_NODES=2
    ;;
  g)
    GPU_MODEL=$OPTARG
    ;;
  r)
    REGION=$OPTARG
    ;;
  h)
    HELP
    ;;
  *)
    echo "UNIMPLEMENTED OPTION -- ${OPTKEY}" >&2
    exit 1
    ;;
  esac
done
shift $(expr $OPTIND - 1)
OTHERARGS=$@

case $GPU_MODEL in
T4)
  GCP_MACHINE="n1-standard-16"
  GCP_GPU_MODEL_NAME="NVIDIA_TESLA_T4"
  ;;
V100)
  GCP_MACHINE="n1-standard-32"
  GCP_GPU_MODEL_NAME="NVIDIA_TESLA_V100"
  ;;
esac

NUM_WORKER_NODES=$((NUM_NODES - 1)) # NUM_NODES includes the master

echo "Submitting Vertex AI PyTorch job with" ${GPU_MODEL} ${CONFIG}

# NUM_GPUS to use (per machine)
NUM_GPUS=4
NUM_WORKERS=4

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=spotlight-perception-models

# Build this using the Dockerfile and the provided script
IMAGE_URI=gcr.io/context-ml/lightningmnist:vertex_ai

# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=lightningmnist_${NUM_NODES}_nodes_x${NUM_GPUS}x${GPU_MODEL}_DDP_$(date +%Y%m%d_%H%M%S)
echo "Monitor job here: https://console.cloud.google.com/ai-platform/jobs/${JOB_NAME}/charts/gpu?project=context-ml"

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/experiments/${JOB_NAME}/

if [[ $NUM_WORKER_NODES -gt 0 ]]; then
  WORKER_SPEC_FLAG="--worker-pool-spec=replica-count=${NUM_WORKER_NODES},machine-type=${GCP_MACHINE},accelerator-type=${GCP_GPU_MODEL_NAME},accelerator-count=${NUM_GPUS},container-image-uri=${IMAGE_URI}"
else
  WORKER_SPEC_FLAG=""
fi

gcloud beta ai custom-jobs create \
  --display-name=${JOB_NAME} \
  --region ${REGION} \
  --enable-web-access \
  --worker-pool-spec=replica-count=1,machine-type=${GCP_MACHINE},accelerator-type=${GCP_GPU_MODEL_NAME},accelerator-count=${NUM_GPUS},container-image-uri=${IMAGE_URI} \
  ${WORKER_SPEC_FLAG} \
  --args="--job-dir=${JOB_DIR}","--trainer.max_epochs=100","--trainer.accelerator='gpu'","--trainer.num_nodes=${NUM_NODES}","--trainer.devices=${NUM_GPUS}","--trainer.strategy='ddp_find_unused_parameters_false'"
