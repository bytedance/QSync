# Predictor for QSync
- Modeling of the local DFG and global DFG with neighboring information (DAG).
- Also, provide statistical guidance (indicator) and status prediction (latency and memory) for distributed mixed-bitwidth training

# Test 
We provide a `test.py` to show the ability of the mixed-bitwdith prediction, also provided sample traces used in the prediction, to use it,
1. `bash trails/resnet50.sh`, and `bash trails/bert.sh`
2. `python3 test.py`
And you will see the results we provided in paper.

### Notice
- Notice that you should disable nn.clipping when doing prediction, since we cannot well-trace these alternative training method.
- Also, when you deal with your own traces, pls run the casting cost again. The result can differ among different instances.

# Structure
- algos.py saves the search method for sync. for the moment, only implemented the greedy method

# Procedure
We only prepare samples for VGG16, ResNet50, Bert, Roberta, but you can follow the same step to produce the prediction and DAG
1. run `fwd_gen.py`, this will generate the DAG with node depth for the model. But make sure that youy have the corresponding input under the input_data folder.
2. Create Traces folder, place the traces that is required for analysis inside
    - Usually the INT8/FP16/FP32 trace
    - for format, pls follow our sample.
3. Run analysis script, follow the scripts inside `trail` folder. e.g. `bash trails/resnet50.sh`
    - this will generate four types of files. `cng` saves the critical path. `cng_data` saves the data on critical path. `node_data` saves mapper-like node data. `dag_graph_folder` saves the modifid DAG graph.
    - all data is savd under the tmp_data

4. Generate casting cost mapper
    - run `python3 casting_cost.py`
    - given the casting cost alpha and beta, update it in the `predictor.py`

5. Test case in the main components. We implement serveral components test case in each file.
    - Cross node prediction result. `python3 cross_node_predict.py` to show multi-node prediction result.
    - Fastest available plan.  `python3 fastest_plan.py` to hierachically derives the fastest mixed-bitwidth plan for a local training.

6. In prediction, we use the togological sort provided by dpro and result will be stored in the .log folder.

7. *Manually run analysis and prediction
    - the entry of the predictor is `analyse.py`, and the command is like
    - `python3 analyse.py --device_name 't4' --bit 32 --model_name $1  --generate_mapping --target_trace_idx 1 --<file_name>`
        - if disable `--generate_mapping` it will direct output the prediction.
    - you can also use the provided `run_single_prepare.sh` and `run_single_predict.sh` to do data generation and prediction
        - like `bash run_single_predict.sh 32 resnet50 fp32_comm_t4` for prediction
        - `bash run_single_prepare.sh 16 resnet50 fp16_comm_t4` to generate node data mappers for a single trace
    - there may exsits some traces that cannot be processed. which results from the missing of some ops, pls change different `trace_idx` to test



# Add new custom kernel
### Add user annotation in LpTorch
use torch.profile to make user annotations in the LP-PyTorch
### Add in Predictor config.
1. add the op's fwd op in `ops_to_track`. e.g. `qsync::linear_fp32`
2. add the aten to bwd mapping. e.g. `'qsync::linear_fp32': ['linearfp32backward']`. Notice to use lower-case for bwd op.
3. if the op is bitwitdh relevant op, add it to `bit_width_changeable_nodes` and `grad_required_bwds`
    - e.g. `'linearint8backward': 2`, the later number is the corresponding accumulateGrad operation numbers required by the bwd op. The number can vary due to your implementation
    - notice only when bias and weight are required then is number 2, else is 1
4. If necessary, add special mapping from the aten kernel to your customized kernel
    - e.g. `'aten::linear': 'qsync::linear_tf32', "qsync::linear_fp32": "qsync::linear_tf32"`.
It do requires some efforts in configuring the configs. Pls carefully check the configurations shown in the `config.py`

    