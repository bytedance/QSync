python3 cross_bit_mem_alter.py --model-name bert --indicator_type GVAR --mem_limit 2422042.5
python3 cross_bit_mem_alter.py --model-name bert --indicator_type WVAR --mem_limit 2422042.5
python3 cross_bit_mem_alter.py --model-name bert --indicator_type HESS --mem_limit 2422042.5

# 2714698.5
# 2452042.5 64% compression 
# 2258506.5

python3 cross_bit_mem_alter.py --model-name roberta --indicator_type GVAR --mem_limit 1150000
python3 cross_bit_mem_alter.py --model-name roberta --indicator_type WVAR --mem_limit 1150000
python3 cross_bit_mem_alter.py --model-name roberta --indicator_type HESS --mem_limit 1150000

# 1373867.5
# 1150000 64% compression 
# 1031723.5