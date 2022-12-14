#!/bin/sh
# replace with your root folder absolute path
DATA_PATH="/data"
ROOT_PATH=$(realpath $(dirname $0))
echo "Qsync project path: $ROOT_PATH"
GPU=0,1,2,3,4,5,6,7 # control the visiable gpus

# deal with the permision problem
docker build -t juntao/qsync:0.3 .

docker run --shm-size=40g --net=host --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPU -v $DATA_PATH:/data -v $ROOT_PATH:/qsync_niti_based -it juntao/qsync:0.3 