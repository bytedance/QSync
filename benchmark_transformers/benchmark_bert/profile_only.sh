accelerate launch --config_file ${HOME}/configs/accelerator/single_node.yaml acc_squad.py \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./output \
  --pad_to_max_length \
  --checkpointing_steps epoch \
  --indicator_type 0 \
  --profile-only 
 