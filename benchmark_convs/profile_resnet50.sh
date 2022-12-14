rm -rf log 
rm -rf .log 
rm -rf logs

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 1

sleep 5
echo "fisnis 1, next"
pkill python3
pkill torchrun

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 2

sleep 5
echo "fisnis 2, next"
pkill python3
pkill torchrun

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 3

sleep 5
echo "fisnis 3, next"
pkill python3
pkill torchrun

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 4

sleep 5
echo "fisnis 4, next"
pkill python3
pkill torchrun

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 5

sleep 5
echo "fisnis 5, next"
pkill python3
pkill torchrun

torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py \
--batch-size 128 --test-batch-size 32 --epochs 90 --model-name RESNET50 --ds-name imagenet --T-B 13 \
 --train-dir /data/ImageNet/train --val-dir /data/ImageNet/val --test-case 6

sleep 5
echo "fisnis 6, next"
pkill python3
pkill torchrun

python3 mv_file.py