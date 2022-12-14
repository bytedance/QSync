# generate the configs for cuda_conv and cuda_linear
from qsync.Utils import get_capability
import os, json 
import torch
import shutil

config_json_path = os.path.abspath(os.path.join(__file__, os.pardir,'qsync', 'LpTorch', "config.json"))


# DMLC_PS_ROOT_URI_PUBLIC="13.213.38.233"
def write_bash_file(file_name, content):
    with open (file_name, 'w') as rsh:
        rsh.write(content)

cutlass_conv_path = os.path.abspath(os.path.join(__file__, os.pardir, "pytorch", "cutlass-conv" , "device_config.h"))
cutlass_linear_path = os.path.abspath(os.path.join(__file__, os.pardir, "pytorch", "cutlass-linear", "device_config.h"))

def generate_backward_conf():
    ver = "1.10" if torch.__version__ == '1.10.0' else "1.12"
    # copy config
    # backward config
    backward_file_path = os.path.abspath(os.path.join(__file__, os.pardir, "pytorch", "other_extension", "backward_func.cc"))
    copy_file_path = os.path.abspath(os.path.join(__file__, os.pardir, "pytorch", "v_temp", ver, "backward_func.cc"))
    shutil.copyfile(copy_file_path, backward_file_path)
    print("copied right backward config to folder")

def generate_cutlass_conf():
    cap = get_capability()
    template = """
    #ifndef device_config
    #define device_config
    {}
    #endif
    """
    if cap == 80:
        config = \
        """
    #include "name_A100.h"
    using namespace sm80_space;
        """
    elif cap == 75:
        config = \
        """
    #include "name_T4.h"
    using namespace sm75_space;
        """
    elif cap == 70:
        config = \
        """
    #include "name_V100.h"
    using namespace sm70_space;
        """

    template = template.format(config)
    write_bash_file(cutlass_conv_path, template)
    write_bash_file(cutlass_linear_path, template)
    print("Generated the configs for cutlass.")

def check_quantization_optimization():
    from qsync.LpTorch.conf import config
    from qsync.LpTorch.quant import quantize_int8

    # in 11.6 quantization optimization is not necessary, original torch has included it. 
    BS_BASE = 64
    BS_TIMES = 6
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    def calculate_duration():
        duration_data = []
        for times in range(1, BS_TIMES):
            A = torch.rand(BS_BASE * times, 64, 56, 56).cuda()
            torch.cuda.synchronize()
            start.record()
            for i in range(100):
                q_A_opt, scale_opt = quantize_int8(A)
            end.record()
            torch.cuda.synchronize()
            Duration = start.elapsed_time(end)
            duration_data.append(Duration)
        return duration_data
    config.optimized_int = True
    duration_opt = calculate_duration()
    config.optimized_int = False
    duration_normal = calculate_duration()
    print("optimzied: ", duration_opt)
    print("not-opt: ",duration_normal)
    cnt = 0
    for idx, val in enumerate(duration_opt):
        if val < duration_normal[idx]:
            cnt += 1
    if cnt / len(duration_opt) >= 0.80:
        print(r"more than 80% optimized latency are faster, use optimized fixed-point quantization")
        config = """
{
    "optimized_int": true
}
"""
        write_bash_file(config_json_path, config)
    else:
        print(r"use default fixed-point quantization")

def generate_target_ops(args):
    json_string = None
    with open(config_json_path, 'r') as jf:
        json_string = json.load(jf)
        json_string['ops'] = args.ops
    with open(config_json_path, 'w') as jf:
        json.dump(json_string, jf)
    print("update target", json_string)

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=' Config of qsync.')
    parser.add_argument('--funct', type=str, default=None, help="function name to run")
    parser.add_argument('--ops', metavar='N', type=str, default=["nn.Conv2d", "nn.ReLU", "nn.BatchNorm2d"], nargs='+',  help='ops to track')
    args = parser.parse_args()
    if args.funct == 'conf':
        generate_cutlass_conf() # generate correct cutlass configuration
        generate_backward_conf()
    elif args.funct == 'qopt':
        check_quantization_optimization()
    elif args.funct == 'tops':
        generate_target_ops(args)