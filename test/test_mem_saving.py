# test the function of conv op
import torch
import torch.nn as nn 
from torchvision import models
from qsync.LpTorch.layers import construct_qconv_by_conv, QConv2d
from qsync.LpTorch.utils import get_memory_usage, compute_tensor_bytes
import copy 
from qsync.LpTorch.layers import replace_old_conv_with_qconv, replace_old_relu_with_qrelu, reset_bits
from QSync import QModule

device = torch.device('cuda')

GB = 1024 * 1024 * 1024

bit = 8
BS = 128

available_bits = [8, 16, 32]

def test_mem_for_bit(inp, bit):
    assert bit in available_bits, "Not available bits"
    init_mem = get_memory_usage(False)

    qconv2d_list = []
    qrelu_list = []
    model = models.resnet50(pretrained=True)
    model = model.cuda()

    qmodel = QModule(model)
    qmodel = qmodel.cuda()

    for n, mod in qmodel.named_modules():
        if isinstance(mod, QConv2d):
            mod.bit = bit


    # op = copy.deepcopy(model.conv1)
    # del model
    model_mem = get_memory_usage(False) 
    print("Model Mem", (model_mem - init_mem) / 1024 / 1024)
    
    data_mem = get_memory_usage(False) 
    print('Data Loader', (data_mem - model_mem) / 1024 / 1024 )
    model.train()
    model_fp = qmodel(inp)
    out_test = torch.rand(BS, 1000, device=device)
    loss_fn = nn.MSELoss()
    loss = loss_fn(out_test, model_fp)
    print("Before BP")
    mem_before_bp = get_memory_usage(True)
    loss.backward()
    print("After BP")
    mem_aft_bp = get_memory_usage(True)

    print(f"{bit}: ActMM + Loss", (mem_before_bp - model_mem) / 1024 / 1024 )

    torch.cuda.empty_cache()

def test_mem_saving():
    inp = torch.rand(BS, 3, 224, 224, device=device) * 2
    for bit in available_bits:
        test_mem_for_bit(inp, bit)
    

