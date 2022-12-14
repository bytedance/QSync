# LP-PyTorch
LP-PyTorch is a low-precision kernel access structure that allows user to directly access the underneath low-precision kernels (e.g. CUTLASS and CUDNN)
- For the current version of LP-PyTorch, we support INT8, FP16 and FP32

We have serveral tags that can be altered in the configuration
- use NCWH or NHWC (since use NCHW requires an extra conversion, INT8 only support NHWC. But BN's support on NHWC is wrong)
- quantization method for activation and kernel. You can use `scale` or `channel` two types, but only `scale + scale`, `scale + channel`, `channel + scale` is allowed.

# Usage
Use of Lp-PyTorch is simple, for any model in PyTorch, use
```python
    model = QModule(model)
```
Then the model is converted to a mixed-bitwith version and supports all the functionality we implemented.
