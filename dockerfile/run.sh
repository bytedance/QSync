#!/bin/sh
DOCKER_BINARY="docker"
IMAGE_NAME="qsync"
CONT_NAME='qsync-test'

# store your data here
DATA_PATH="/data"
echo "The value of DATA_PATH is: $DATA_PATH"
# ROOT_PATH=$(realpath $(dirname $0))
ROOT_PATH=$(realpath "$(dirname "$0")/..")

# MOUNT_PATH=$(<mount_path.txt)
# echo "The value of MOUNT_PATH (QSYNC_ROOT) is: $MOUNT_PATH"

${DOCKER_BINARY} run --ipc host --gpus all -v $DATA_PATH:/data -v $ROOT_PATH:/qsync --name ${CONT_NAME} -it ${IMAGE_NAME}