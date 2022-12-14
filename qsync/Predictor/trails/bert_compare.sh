bash run_single_prepare.sh bert bert_bs12_t4
bash run_single_prepare.sh bert bert_bs24_v100
python3 cross_bit_alter.py --model-name bert --backbone bert_bs12_t4 --comparer bert_bs24_v100