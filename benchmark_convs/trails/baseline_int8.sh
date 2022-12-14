export QSYNC_SIMU=1
export IP=<YOURIP>
export PORT=4322
export RANK=0
export NUM_SERVERS=4
export WORKERS_PER_SERVER=8
export DATA_PATH=<YOURPATH>
export START_LR=4.096
export MODEL_NAME=RESNET50


torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--batch-size 128 --dy-bs 128 --lr $START_LR --test-batch-size 32 --epochs 120 --model-name $MODEL_NAME --ds-name imagenet \
--train-dir $DATA_PATH/train --val-dir $DATA_PATH/val



export QSYNC_SIMU=1
export IP=<YOURIP>
export PORT=4322
export RANK=1
export NUM_SERVERS=4
export WORKERS_PER_SERVER=8
export DATA_PATH=<YOURPATH>
export START_LR=4.096
export MODEL_NAME=RESNET50


torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--batch-size 128 --dy-bs 128 --lr $START_LR --test-batch-size 32 --epochs 120 --model-name $MODEL_NAME --ds-name imagenet \
--train-dir $DATA_PATH/train --val-dir $DATA_PATH/val


export QSYNC_SIMU=1
export IP=<YOURIP>
export PORT=4322
export RANK=2
export NUM_SERVERS=4
export WORKERS_PER_SERVER=8
export DATA_PATH=<YOURPATH>
export START_LR=4.096
export MODEL_NAME=RESNET50


torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--batch-size 128 --dy-bs 128 --lr $START_LR --test-batch-size 32 --epochs 120 --model-name $MODEL_NAME --ds-name imagenet \
--train-dir $DATA_PATH/train --val-dir $DATA_PATH/val --int8

export QSYNC_SIMU=1
export IP=<YOURIP>
export PORT=4322
export RANK=3
export NUM_SERVERS=4
export WORKERS_PER_SERVER=8
export DATA_PATH=<YOURPATH>
export START_LR=4.096
export MODEL_NAME=RESNET50


torchrun --nproc_per_node=$WORKERS_PER_SERVER --nnodes=$NUM_SERVERS --node_rank=$RANK --master_addr=$IP --master_port=$PORT main.py \
--batch-size 128 --dy-bs 128 --lr $START_LR --test-batch-size 32 --epochs 120 --model-name $MODEL_NAME --ds-name imagenet \
--train-dir $DATA_PATH/train --val-dir $DATA_PATH/val --int8
