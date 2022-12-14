accelerate launch --config_file ${HOME}/configs/accelerator/single_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=16 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --output_dir ./output \
  --indicator_type 0 \
  --profile-only 

  