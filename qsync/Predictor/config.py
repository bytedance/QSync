import os
import torch 
assert torch.__version__ >= '1.10.0', "Require Torch > 1.10.0"
LASTEST_TORCH = torch.__version__ > '1.11.0'

KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3

class PredictorConfig:
    def __init__(self, CONSIDER_OPTIMIZER_TIME=False):
        self.CONSIDER_OPTIMIZER_TIME=CONSIDER_OPTIMIZER_TIME
        self.pred_root = os.path.join(__file__, os.pardir)
        self.trace_folder = os.path.join(__file__, os.pardir, 'traces', 'exp')
        self.dag_folder = os.path.join(__file__, os.pardir, 'tmp_data', "dag_graph_folder")
        self.cng_folder = os.path.join(__file__, os.pardir, 'tmp_data', "cng")
        self.cng_data_folder = os.path.join(__file__, os.pardir, 'tmp_data', "cng_data")
        self.node_data_folder = os.path.join(__file__, os.pardir, 'tmp_data', "node_data")
        self.mapper_result_folder = os.path.join(__file__, os.pardir, 'tmp_data', "mapper_result")
        self.root_node_name = 'root' # node used in graph construction
        self.pre_root_node_name = 'root_pre' # once consider the previous node, required

        self.comm_bias = 13

        self.enable_cast_cost_modeling = True # turn off it to show the difference
        self.enable_test_gap = False # turn on to show the difference in CPU step estimation
        self.fastest_plan_iso = True # use isomorphic to accelerate the fastest plan searching, most case is correct, but may produce wrong result.

        self.bit16_mapping, self.bit32_mapping = None, None # need assignment

        self.KB = KB
        self.MB = MB
        self.GB = GB


        self.cpu_time_gaps = {
            't4':
            {
                'fwd': 0.02, # 0.02 for int8, 0.06 for half
                'bwd': 0.005,
                'optimizer_step': 0.02,
                'optimizer_zero_grad': 0.01
            },
            'v100':
            {
                'fwd': 0.06,
                'bwd': 0.005,
                'optimizer_step': 0.04,
                'optimizer_zero_grad': 0.02
            }
        }

        # enable to show the difference
        self.cpu_test_gap = {
            't4':
            {
                'fwd': 0.01, 
                'bwd': 0.01,
                'optimizer_step': 0.01,
                'optimizer_zero_grad':0.01
            },
            'v100':
            {
                'fwd': 0.04,
                'bwd': 0.04,
                'optimizer_step': 0.04,
                'optimizer_zero_grad': 0.04
            }
        }

        self.output_bit_mapping = {
            8: 32,
            16: 16,
            32: 32
        }

        self.output_bit_mapping_bwd = {
            8: 16,
            16: 16,
            32: 32
        }

        self.cur_cpu_gaps = None 


        self.LASTEST_TORCH = LASTEST_TORCH

        # runtime ops that is useless
        self.useless_runtime_name = [
            'cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags', 'cudaStreamGetCaptureInfo', 'cudaEventQuery', 'cudaStreamIsCapturing',
            'cudaMalloc', 'cudaStreamWaitEvent', 'INVALID', 'cudaDeviceGetAttribute', 'cudaPeekAtLastError', 'cudaStreamSynchronize', 'cudaFuncSetAttribute',
            'cudaEventRecord', "cudaPointerGetAttributes", 'cudaFuncGetAttributes', 'cudaEventCreateWithFlags', 'cudaEventDestroy'

        ]


        # operators to be tracked
        # since cuda luanch takes most of the time, the operators that incurs cuda launch is the one used for tracing
        self.ops_to_track = [
            'aten::embedding', 'aten::add', 'aten::add_', 'aten::layer_norm', 'aten::dropout', 'aten::cat', 'aten::rsub', 'aten::mul',
            'aten::linear', 'aten::matmul', 'aten::div', 'aten::div_', 'aten::softmax', 'aten::contiguous', 'aten::copy_', 'aten::clamp',
            'aten::gelu', 'aten::log_softmax', 'aten::nll_loss_nd', 'aten::view', 'aten::permute', 'aten::transpose', 'aten::split', 'aten::squeeze',
            "aten::flatten_dense_tensors", "aten::unflatten_dense_tensors", "aten::unsqueeze", "aten::empty",
            # convs
            'aten::flatten', 'aten::conv2d','aten::tanh', 'aten::select', 'aten::slice', 'aten::pow', "aten::mean", "aten::as_strided_",
            "aten::native_batch_norm", "aten::cudnn_batch_norm", 'aten::relu', 'aten::zeros', 'aten::zero_', 'aten::max_pool2d_with_indices',
            "aten::ones_like", 'aten::detach_', 'aten::relu_', 'aten::adaptive_avg_pool2d',
            # cast related
            'aten::_to_copy',
            # tensorecore related
            'aten::constant_pad_nd', 
            # int8 forced synchronization
            'aten::item',
            # self-defined
            "aten::quantize_int8", 'qsync::cast_cost_fp16',
            'qsync::linear_fp32', 'qsync::linear_fp16', 'qsync::linear_int8', 'qsync::linear_tf32', 
            'qsync::conv_fp32', 'qsync::conv_fp16', 'qsync::conv_int8', 'qsync::conv_tf32', 
            # fwd nccl call
            "nccl:broadcast",
            # act nn
            "act::relu",
            # gradient clipping
            "aten::norm", "aten::detach"
        ]

        self.casting_ops = ["aten::quantize_int8", 'qsync::cast_cost_fp16']

        self.aten_op_to_bwd_mapper = {
            'aten::clamp': [],
            'aten::tanh': [],
            'aten::log_softmax': ['logsoftmaxbackward0'],
            'aten::embedding': ['embeddingbackward0'],
            'aten::div': ['divbackward0'],
            'aten::div_': ['divbackward0'],
            'aten::matmul': ['bmmbackward0'],
            'aten::layer_norm': ['nativelayernormbackward0'],
            'aten::gelu': ['gelubackward0'],
            'aten::add': ['addbackward0'],
            'aten::add_': ['addbackward0'],
            'aten::nll_loss_nd': ['nlllossbackward0'],
            'aten::softmax': ['softmaxbackward0'],
            'aten::dropout': ['nativedropoutbackward0' if LASTEST_TORCH else 'fuseddropoutbackward0'],
            # for aten::linnear, there exists two case. In F.linear, admmbackward thus is partitioned as AddBackward0,  MmBackward0 
            # which means the corresponding bp function should be 'addmmbackward0', and AddBackward0,  MmBackward0 
            'aten::linear': ['addmmbackward0', 'viewbackward0', 'tbackward0'] if LASTEST_TORCH else ['addbackward0', 'mmbackward0', 'viewbackward0', 'tbackward0'], 
            'aten::transpose': ['transposebackward0'],
            'aten::view':['viewbackward0'],
            'aten::permute':['permutebackward0'],
            'aten::contiguous': ['clonebackward0'],
            'aten::split': ['splitbackward0'],
            'aten::squeeze': ['squeezebackward1'],
            'aten::slice': ['slicebackward0'],
            'aten::select': ['selectbackward0'],
            'aten::pow': ['powbackward'],
            "aten::mul": ["mulbackward"],
            'aten::flatten': ['reshapealiasbackward0'], # flatten calls reshape_alias,
            'aten::relu': ['relubackward0'],
            'aten::adaptive_avg_pool2d': ['adaptiveavgpool2dbackward0'], # call mean and as strided
            'aten::conv2d': ['aten::cudnn_convolution_backward'],
            'aten::native_batch_norm': ['nativebatchnormbackward0'],
            'aten::cudnn_batch_norm': ['cudnnbatchnormbackward0'],
            'aten::_to_copy': ['tocopybackward0'],
            'aten::max_pool2d_with_indices': ['maxpool2dwithindicesbackward0'],
            'aten::relu': ['relubackward0'],
            'aten::relu_': ['relubackward0'],
            "act::relu": ['torch::autograd::cppnode<actquantizedrelu>'],
            'aten::as_strided_': ['asstridedbackward0'],
            'aten::mean': ['meanbackward1'],
            # customized op
            'qsync::linear_fp32': ['linearfp32backward'],
            'qsync::linear_tf32': ['lineartf32backward'],
            'qsync::linear_fp16': ['linearfp16backward'],
            'qsync::linear_int8': ['linearint8backward'],
            'qsync::conv_fp32': ['conv2dfp32backward'],
            'qsync::conv_tf32': ['conv2dtf32backward'],
            'qsync::conv_fp16': ['conv2dfp16backward'],
            'qsync::conv_int8': ['conv2dint8backward'],
        }




        self.bias_handling = [
            'linearfp32backward', 'lineartf32backward', 'linearfp16backward', 'linearint8backward',
            'conv2dfp32backward', 'conv2dtf32backward', 'conv2dfp16backward', 'conv2dint8backward',
        ]

        # in bert, only these ops requires grad
        self.grad_required_bwds = {
            'addmmbackward0': 2,
            'mmbackward0': 1,
            'nativelayernormbackward0': 2,
            'embeddingbackward0': 1,
            'addbackward0': 1,
            # self-defined kernels. Respect to one kernel
            'conv2dint8backward': 1, 
            'conv2dfp16backward': 1, 
            'conv2dfp32backward': 1, 
            'conv2dtf32backward': 1, 

            'linearint8backward': 1, 
            'linearfp16backward': 1, 
            'linearfp32backward': 1, 
            'lineartf32backward': 1, 
            
            "nativebatchnormbackward0":2, 
            "cudnnbatchnormbackward0":2,
        }



        self.next_mm_target = ['addmmbackward0', 'embeddingbackward0'] + \
         ['linearfp32backward', 'lineartf32backward', 'linearfp16backward', 'linearint8backward'] + \
         ['conv2dfp32backward', 'conv2dtf32backward', 'conv2dfp16backward', 'conv2dint8backward'] +\
         ['fuseddropoutbackward0', 'asstridedbackward0',  'cudnnbatchnormbackward0']

        # when map node on fwd DAG graph to the node in the critical path graph. 
        # here, we want the linear node mapped to our customized linear node
        self.special_aten_mapping = {
            'aten::linear': ['qsync::linear_fp32', 'qsync::linear_tf32', 'qsync::linear_fp16', 'qsync::linear_int8'],
            'aten::conv2d': ['qsync::conv_fp32', 'qsync::conv_tf32', 'qsync::conv_fp16', 'qsync::conv_int8'],
            'batch_norm': ['aten::batch_norm', 'aten::cudnn_batch_norm', 'aten::native_batch_norm'],
            'aten::relu': ['aten::relu', 'aten::relu_', 'act::relu'],
            'aten::max_pool2d': ['aten::max_pool2d_with_indices']
        }
        # DAG to Node mapping
        self.int8_name_mapping = {
            'aten::linear': 'qsync::linear_int8', "qsync::linear_tf32": "qsync::linear_int8", "qsync::linear_fp32": "qsync::linear_int8", "qsync::linear_fp16": "qsync::linear_int8",
            'LinearFP32Backward': 'LinearInt8Backward', 'LinearTF32Backward': 'LinearInt8Backward', 'LinearFP16Backward': 'LinearInt8Backward',
            'aten::conv2d': 'qsync::conv_int8', "qsync::conv_tf32": "qsync::conv_int8", "qsync::conv_fp32": "qsync::conv_int8", "qsync::conv_fp16": "qsync::conv_int8",
            'Conv2dFP32Backward': 'Conv2dInt8Backward', 'Conv2dTF32Backward': 'Conv2dInt8Backward', 'Conv2dFP16Backward': 'Conv2dInt8Backward',
        }

        

        self.fp16_name_mapping = {
            'aten::linear': 'qsync::linear_fp16', "qsync::linear_tf32": "qsync::linear_fp16", "qsync::linear_fp32": "qsync::linear_fp16", "qsync::linear_int8": "qsync::linear_fp16",
            'LinearFP32Backward': 'LinearFP16Backward', 'LinearTF32Backward': 'LinearFP16Backward', 'LinearInt8Backward': 'LinearFP16Backward',
            'aten::conv2d': 'qsync::conv_fp16', "qsync::conv_tf32": "qsync::conv_fp16", "qsync::conv_fp32": "qsync::conv_fp16", "qsync::conv_int8": "qsync::conv_fp16",
            'Conv2dFP32Backward': 'Conv2dFP16Backward', 'Conv2dTF32Backward': 'Conv2dFP16Backward', 'Conv2dInt8Backward': 'Conv2dFP16Backward',
            
        }

        self.fp32_name_mapping = {
            'aten::linear': 'qsync::linear_fp32', "qsync::linear_tf32": "qsync::linear_fp32",  "qsync::linear_fp16": "qsync::linear_fp32", "qsync::linear_int8": "qsync::linear_fp32",
            'LinearTF32Backward': 'LinearFP32Backward', 'LinearFP16Backward': 'LinearFP32Backward', 'LinearInt8Backward': 'LinearFP32Backward',
            'aten::conv2d': 'qsync::conv_fp32', "qsync::conv_tf32": "qsync::conv_fp32", "qsync::conv_fp16": "qsync::conv_fp32", "qsync::conv_int8": "qsync::conv_fp32",
            'Conv2dTF32Backward': 'Conv2dFP32Backward', 'Conv2dFP16Backward': 'Conv2dFP32Backward', 'Conv2dInt8Backward': 'Conv2dFP32Backward'
        }

        self.tf32_name_mapping = {
            'aten::linear': 'qsync::linear_tf32', "qsync::linear_fp32": "qsync::linear_tf32",  "qsync::linear_fp16": "qsync::linear_tf32", "qsync::linear_int8": "qsync::linear_tf32",
            'LinearFP32Backward': 'LinearTF32Backward', 'LinearFP16Backward': 'LinearTF32Backward', 'LinearInt8Backward': 'LinearTF32Backward',
            'aten::conv2d': 'qsync::conv_tf32', "qsync::conv_fp32": "qsync::conv_tf32", "qsync::conv_fp16": "qsync::conv_tf32", "qsync::conv_int8": "qsync::conv_tf32",
            'Conv2dFP32Backward': 'Conv2dTF32Backward', 'Conv2dFP16Backward': 'Conv2dTF32Backward', 'Conv2dInt8Backward': 'Conv2dTF32Backward'
        }


        self.mem_relevant_op = [
            'linear', 'conv', 'matmul', 'gelu', 'softmax', 'layernorm', 'batchnorm', 'layer_norm', 'batch_norm', 'embedding'
        ]
        self.implement_limit_ops = [
            'matmul', 'gelu', 'softmax', 'layernorm', 'batchnorm', 'layer_norm', 'batch_norm', 'embedding'
        ]


        self.allow_list = [
            'linear', 'conv', 'matmul'
        ]

        # correspodnign fwd bitwidth kernels: to allow_lists
        self.bit_width_changeable_nodes = [
            'aten::embedding', 'aten::linear', 'aten::conv2d', 
            'qsync::linear_fp32', 'qsync::linear_fp16', 'qsync::linear_int8', 
            'qsync::conv_fp32', 'qsync::conv_fp16', 'qsync::conv_int8',
            'aten::matmul',
        ]
        self.bit_width_changeable_dag_nodes = [
            'linear', 'embedding', 'conv'
        ]
        # TODO: only handled for two types of the models, may be more 
        self.cast_required_nodes = [
            'aten::layer_norm', 'aten::native_batch_norm', 'aten::cudnn_batch_norm', 'batch_norm', "aten::gelu", 'aten::softmax',
        ] + self.bit_width_changeable_nodes

        self.deny_list = [
            'layernorm', 'batchnorm', 'layer_norm', 'batch_norm', 'gelu', 'softmax', 'aten::batch_norm', 'batch_norm', 'add',
            'subgrapph_c' # subgraph concate node
        ]

        self.deny_nodes = [
            'aten::layer_norm', 'aten::native_batch_norm', 'aten::cudnn_batch_norm', 'aten::batch_norm', 'batch_norm', 'aten::gelu', 'aten::softmax', 
            'aten::log_softmax','aten::nll_loss_nd'
        ]


        self.quantize_fused = False # whehter fused quantize in fwd pass
        self.act_quant = 'scale'
        self.kernel_quant = 'scale'

        # communication pack for different training, collect through torch communication hook
        # not necessary correct.
        self.comm_pack_size = {
            'bert': [2363138, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872,  7087872,  28563456],
            'roberta': [591361, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872, 7087872,  7087872,  39000576],
            'resnet50': [2049000, 7875584, 6563840, 6637568, 2431040],
            'vgg16': [4097000, 16781312, 102764544, 7079424, 7079936, 555328]
        }

        self.cpu_interference_cost = 40

        self.inference_prediction = False # turn on the inference prediction of layers. TODO: implement in future


        


    
    def set_cur_cpu_gaps(self, device):
        self.cur_cpu_gaps = self.cpu_time_gaps[device]
    
    def set_fwd_time(self, cpu_time):
        self.cur_cpu_gaps['fwd'] = cpu_time
    
    def set_bwd_time(self, cpu_time):
        self.cur_cpu_gaps['bwd'] = cpu_time
    
    def set_gaps_with_config(self, gap_config):
        for k, v in gap_config.items():
            if k in self.cur_cpu_gaps:
                self.cur_cpu_gaps[k] = v
    
    def set_bitwidth_mapping(self, bit8_mapping=None, bit16_mapping=None, bit19_mapping=None, bit32_mapping=None):
        self.bit8_mapping = bit8_mapping
        self.bit16_mapping = bit16_mapping
        self.bit19_mapping = bit19_mapping
        self.bit32_mapping = bit32_mapping

    
    def set_bitwidth_cast_mapper(self, bit8_cast_mapper=None, bit16_cast_mapper=None, bit19_cast_mapper=None, bit32_cast_mapper=None):
        self.bit8_cast_mapper = bit8_cast_mapper
        self.bit16_cast_mapper = bit16_cast_mapper
        self.bit19_cast_mapper = bit19_cast_mapper
        self.bit32_cast_mapper = bit32_cast_mapper
    
    def get_mapper_with_bit(self, bit):
        if bit == 8:
            return self.bit8_mapping, self.int8_name_mapping
        elif bit == 16:
            return self.bit16_mapping, self.fp16_name_mapping
        elif bit == 19:
            return self.bit19_mapping, self.tf32_name_mapping
        elif bit == 32:
            return self.bit32_mapping, self.fp32_name_mapping
    

config = PredictorConfig()



