# prepare the necessary data for bit cross testing
python3 fwd_gen.py --model-name $1
# generate mappers
python3 analyse.py --device_name 't4' --bit 16 --model_name $1 --generate_mapping --target_trace_idx 1
python3 analyse.py --device_name 't4' --bit 8 --model_name $1  --generate_mapping --target_trace_idx 1
# generate backbone and ref
python3 analyse.py --device_name 't4' --bit 32 --model_name $1  --generate_mapping --target_trace_idx 1
# python3 analyse.py --device_name 'v100' --bit 32 --model_name $1  --generate_mapping
python3 analyse.py --device_name 'v100' --bit 32 --model_name $1  --generate_mapping --target_trace_idx 1
# Resnet50
# python3 analyse.py --device_name 't4' --bit 16 --model_name 'resnet50'  --target_trace_idx 1 --generate_mapping 
# python3 analyse.py --device_name 't4' --bit 8 --model_name 'resnet50'  --target_trace_idx 1 --generate_mapping 
# # generate backbone and ref
# python3 analyse.py --device_name 't4' --bit 32 --model_name 'resnet50'  --generate_mapping 
# python3 analyse.py --device_name 'v100' --bit 32 --model_name 'resnet50'  --generate_mapping
# python3 analyse.py --device_name 'v100' --bit 32 --model_name 'bert'  --generate_mapping




# python3 analyse.py --device_name 'v100' --bit 16 --model_name 'roberta' --generate_mapping