# V100
accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=32 \
  --dy-bs 24 \
  --learning_rate 7.5e-5 \
  --num_train_epochs 6 \
  --seed 42 \
  --output_dir ./output \
  --pad_to_max_length  


# T4
accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=32 \
  --dy-bs 8 \
  --learning_rate 7.5e-5 \
  --num_train_epochs 6 \
  --seed 42 \
  --output_dir ./output \
  --pad_to_max_length  
