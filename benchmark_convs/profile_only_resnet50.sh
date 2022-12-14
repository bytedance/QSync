DATA_PATH=/data/ImageNet
export QSYNC_SIMU=0
python3 main.py \
--batch-size 64 --dy-bs 64 --test-batch-size 32 --epochs 120 --model-name RESNET50 --ds-name imagenet \
 --train-dir ${DATA_PATH}/train --val-dir ${DATA_PATH}/val --profile-only --indicator_type 2