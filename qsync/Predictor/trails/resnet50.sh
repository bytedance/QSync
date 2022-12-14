MODELNAME='resnet50'
bash run_prepare.sh $MODELNAME

python3 analyse.py --device_name 't4' --bit 16 --model_name 'resnet50' --file_name fp16_conv_comm_t4 --target_trace_idx 0
python3 analyse.py --device_name 't4' --bit 8 --model_name 'resnet50' --file_name int8_conv_comm_t4 --target_trace_idx 0
python3 analyse.py --device_name 't4' --bit 16 --model_name 'resnet50' --file_name fp16_layer2_comm_t4 --target_trace_idx 0
