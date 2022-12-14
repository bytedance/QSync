# Syncer
Synchronize layers latency
Use
```bash
    python3 cross_bit_alter.py --model-name resnet50
```
# Notice
Make sure your backbone trace and target trace both have communication, elsewise it will end with a `no end_of_train` error. 

# Structure
- `mem_ilp.py` searches the bitwidth plan if there is no latency constraints. 