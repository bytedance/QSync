accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_squad.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 24 \
  --dy-bs 12 \
  --learning_rate 1.2e-4 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output \
  --seed 42 \
  --pad_to_max_length  

accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_squad.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 24 \
  --dy-bs 12 \
  --learning_rate 1.2e-4 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output \
  --seed 62 \
  --pad_to_max_length  \
  --mapper_path /opt/tiger/launch/qsync/bert_GVAR_bitplan_0.7.npy

accelerate launch --config_file ${HOME}/configs/accelerator/multi_node.yaml acc_squad.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 24 \
  --dy-bs 12 \
  --learning_rate 1.2e-4 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output \
  --seed 42 \
  --pad_to_max_length  \
  --tbitwidth 8