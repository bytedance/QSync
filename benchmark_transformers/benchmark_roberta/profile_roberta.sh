rm -rf log 
rm -rf .log 
rm -rf logs

# data
# accelerate launch generate_input_data.py \
#   --model_name_or_path bert-base-uncased \
#   --dataset_name squad \
#   --per_device_train_batch_size 12 \
#   --learning_rate 3e-5 \
#   --num_train_epochs 2 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ./output \
#   --pad_to_max_length \
#   --checkpointing_steps epoch 
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
export DATASET_NAME=swag
# generate config: necessary
# accelerate config

accelerate launch --config_file ${HOME}/configs/accelerator/single_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --test-case 1

sleep 5
echo "fisnis 1, next"
pkill python3
pkill torchrun

accelerate launch --config_file ${HOME}/configs/accelerator/single_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --test-case 2

sleep 5
echo "fisnis 2, next"
pkill python3
pkill torchrun

accelerate launch --config_file ${HOME}/configs/accelerator/single_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --test-case 3

sleep 5
echo "fisnis 3, next"
pkill python3
pkill torchrun

accelerate launch --config_file ${HOME}/configs/accelerator/single_node_multi_gpus.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --test-case 8

sleep 5
echo "fisnis 8, next"
pkill python3
pkill torchrun

# accelerate launch --config_file ${HOME}/configs/accelerator/single_node_multi_gpus.yaml acc_swag.py \
#   --model_name_or_path roberta-base \
#   --dataset_name swag \
#   --per_device_eval_batch_size=16 \
#   --per_device_train_batch_size=16 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./output \
#   --test-case 5

# sleep 5
# echo "fisnis 5, next"
# pkill python3
# pkill torchrun

# accelerate launch  acc_swag.py \
#   --model_name_or_path roberta-base \
#   --dataset_name swag \
#   --per_device_eval_batch_size=16 \
#   --per_device_train_batch_size=16 \
#   --learning_rate 5e-5 \
#   --num_train_epochs 3 \
#   --output_dir ./output \
#   --test-case 6

# sleep 5
# echo "fisnis 6, next"
# pkill python3
# pkill torchrun

python3 mv_file.py --model-name 'roberta'