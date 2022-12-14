MODELNAME='bert'
bash run_prepare.sh $MODELNAME

python3 analyse.py --device_name 't4' --bit 16 --model_name 'bert' --file_name fp16_li_comm_t4 --target_trace_idx 1
python3 analyse.py --device_name 't4' --bit 8 --model_name 'bert' --file_name int8_li_comm_t4 --target_trace_idx 0
python3 analyse.py --device_name 't4' --bit 16 --model_name 'bert' --file_name fp16_layer3_comm_t4 --target_trace_idx 0
python3 analyse.py --device_name 't4' --bit 16 --model_name 'bert' --file_name mixed_arch75 --target_trace_idx 0
