# QSync
Official resporitory for "IPDPS' 24 QSync: Quantization-Minimized Synchronous Distributed Training Across Hybrid Devices".

### Description
QSync aims to explore the potential of removing unnecessary quantized operations to improve training accuracy. It achieves this through the following components:
- Quantization perturbation indicator/Replayer for analyzing the global data flow graph's memory and latency under mixed-precision (Predictor)
- Allocator for selecting the optimal quantized operations for training (Allocator / Syncer)
- Support for low-precision backends (CUTLASS, CUDNN) (LP-PyTorch)
In particular, QSync addresses a specific practical scenario: hybrid-cluster training, which involves an inference GPU with power capabilities (memory, compute) and a training GPU with higher capabilities.

The provided scripts support both convolution-based and transformer-based models.

*NOTE*: The project is a bit old. The performance of kernel implementation may not catch up with latest PyTorch.

### Set Environment
Clone the repo
`git clone --recursive https://github.com/bytedance/QSync.git`

#### Docker
- run `build.sh` under `dockerfile`
- run `run.sh`, specifiying the necessary path mounting inside.
- run `pip install -e .` right in the root folder of QSync, compilation of kernels will start.

#### Manual Installation
- Some libs may hard to install without proxy. Change `<abspath_to_root>` in `m_install.sh` to the absolute path to the root folder. Then
1. `bash m_install.sh`
2. `make`

### Usage
QSync is implemented under the `qsync` folder, composed of `syncer`, `predictor` and `LpTorch`.
- to use LpTorch and convert your model to mixed-biwdith model, use `model = QModule(model)`
- See detail for usage of predictor and syncer in the corresponding page.
- See sample under `benchmark_convs / benchmark_transformers`

*notice* the cross-node cost modeling is not as accurate as single-node is. Extra efforts required to align the communication start.



