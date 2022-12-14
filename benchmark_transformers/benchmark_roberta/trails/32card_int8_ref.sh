# QSync
accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_swag.py \
  --model_name_or_path roberta-base \
  --dataset_name swag \
  --per_device_eval_batch_size=16 \
  --per_device_train_batch_size=32 \
  --dy-bs 16 \
  --learning_rate 7.5e-5 \
  --num_train_epochs 6 \
  --seed 62 \
  --output_dir ./output \
  --pad_to_max_length  

accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_swag.py \
--model_name_or_path roberta-base \
--dataset_name swag \
--per_device_eval_batch_size=16 \
--per_device_train_batch_size=32 \
--dy-bs 16 \
--learning_rate 7.5e-5 \
--num_train_epochs 6 \
--seed 62 \
--output_dir ./output \
--pad_to_max_length \
--mapper_path /opt/tiger/launch/qsync/roberta_GVAR_bitplan_0.7.npy



accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_swag.py \
--model_name_or_path roberta-base \
--dataset_name swag \
--per_device_eval_batch_size=16 \
--per_device_train_batch_size=32 \
--dy-bs 16 \
--learning_rate 7.5e-5 \
--num_train_epochs 6 \
--seed 62 \
--output_dir ./output \
--pad_to_max_length \
--tbitwidth 8
# DB
