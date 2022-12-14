# accelerate config
export DATASET_NAME=swag
export NVIDIA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file ${PROJECT_DIR}/configs/accelerator/single_node.yaml generate_input_data.py \
  --model_name_or_path roberta-base \
  --dataset_name $DATASET_NAME \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output
