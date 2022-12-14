# QSync
Official resporitory for "QSync: Adpative Mixed-Precision for Training Synchronization". Paper to be released in future.
NB: Still an ongoing work.

### Description
QSync tries to exploit the potentials in using quantization for compression-aware training acceleration.
- QSync fully exploits the flexibility and generality of lower-bit kernels ([CUTLASS](https://github.com/NVIDIA/cutlass), CUDNN) for acceleration and reduce the overhead by several optimizations (LP-PyTorch)
- Model the data flow graph for the multi-device mixed-bitwidth case, accurately predict the latency and memory occupation for the single-device and multi-device case (Predictor)
- And finally did the compression-minimzed low-precision kernel plan based on the above two components. Achieving the degradation mimized training acceleration. (Syncer)


For the current version, conv-based and transformer-based model are supported.

### Set Environment
#### Docker
Recommend to use docker. Type
`bash docker_run_mount.sh`
in an ec2 device with 8 gpus, you can start a qsync environment (modify the docker_run_mount.sh file to change the number of gpus). then, run
`make`
to create the qsync environment. 
#### Manual Installation
Change `<abspath_to_root>` in `manual_install.sh` to the absolute path to the root folder. Then
`bash manual_install.sh`
Then
`make`

### Usage
QSync is implemented under the `qsync` folder, composed of `syncer`, `predictor` and `LpTorch`.
- to use LpTorch and convert your model to mixed-biwdith model, use `model = QModule(model)`
- See detail for usage of predictor and syncer in the corresponding page.
- See sample under `benchmark_convs / benchmark_transformers`



