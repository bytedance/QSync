bash run_single_prepare.sh roberta roberta_bs16_t4
bash run_single_prepare.sh roberta roberta_bs32_v100
python3 cross_bit_alter.py --model-name roberta --backbone roberta_bs16_t4 --comparer roberta_bs32_v100