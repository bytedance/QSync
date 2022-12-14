'''
    The profiling result for Dyanmic Batching.
    Dynamic batch can hardly estimate the latency for the batchsize change, But 
    It can acquire a relative correct memory increment regarding the bs with respect to *any device
    The memory usage in the cluster will be:
    : model_occpuation: modelsize + bs * bs_mem / per_bs + Pytorch CUDA context and CuDNN kernel reservation.
    - So basically we can estimate the memory change by bs_mem / per_bs

    The data collected here may not consistent to torch since we use our handwritten layer & model
    - also, model used in CIFAR can be different from the Imagenet Due to version problem
'''
KB = 1024 
GB = 1024 ** 3
# 128, 64, 32, 1(model size)
# collected though torch.cuda.memory_allocated(device)
VGG_16_MEM_CIFAR = [323237888, 251144704, 215361024, 2306955520]
RESNET18_MEM_CIFAR = [453620224, 295281664, 216244224, 1394145280]
RESNET50_MEM_CIFAR = [2050367488, 1171657216, 731909632, 3048555520]

# 12, 8, 4, 1
# for transfomer, the input is dynamic, take the maximum memory profiled
BERT_MEM = [7929689600, 5845669376, 3847501312, 1967926784 + 1 * GB]
ROBERTA_MEM = [2050367488, 1171657216, 731909632, 3048555520]


VGG_16_MEM_IMAGENET = [12386525 * KB, 6931987 * KB, 4210878 * KB, 2306955520]
RESNET50_MEM_IMAGENET = [19191384 * KB, 9727778 * KB, 4970695 * KB, 3048555520]
def cal_mem_scale(mem_data):
    mem_scale = (mem_data[0] - mem_data[1]) / 64 // KB
    mem_scale2 = (mem_data[1] - mem_data[2]) / 32 // KB
    # return mem_scale, mem_scale2
    return max(mem_scale, mem_scale2), mem_data[3] // KB

def cal_mem_scale_trans(mem_data):
    mem_scale = (mem_data[0] - mem_data[1]) / 4 // KB
    mem_scale2 = (mem_data[1] - mem_data[2]) / 4 // KB
    # return mem_scale, mem_scale2
    # print(mem_scale, mem_scale2)
    return max(mem_scale, mem_scale2), mem_data[3] // KB
# print(cal_mem_scale(RESNET50_MEM_CIFAR))

sta_result_cifar = {
    'VGG16': {
        'model_size': cal_mem_scale(VGG_16_MEM_CIFAR)[1],
        'scale_mem_bs': cal_mem_scale(VGG_16_MEM_CIFAR)[0],
    },
    'RESNET18': {
        'model_size': cal_mem_scale(RESNET18_MEM_CIFAR)[1],
        'scale_mem_bs': cal_mem_scale(RESNET18_MEM_CIFAR)[0],
    },
    'RESNET50': {
        'model_size': cal_mem_scale(RESNET50_MEM_CIFAR)[1],
        'scale_mem_bs': cal_mem_scale(RESNET50_MEM_CIFAR)[0],
    }
}


sta_result_imagenet = {
    'VGG16': {
        'model_size': cal_mem_scale(VGG_16_MEM_IMAGENET)[1],
        'scale_mem_bs': cal_mem_scale(VGG_16_MEM_IMAGENET)[0],
    },
    'RESNET18': {
        'model_size': cal_mem_scale(RESNET18_MEM_CIFAR)[1],
        'scale_mem_bs': cal_mem_scale(RESNET18_MEM_CIFAR)[0],
    },
    'RESNET50': {
        'model_size': cal_mem_scale(RESNET50_MEM_IMAGENET)[1],
        'scale_mem_bs': cal_mem_scale(RESNET50_MEM_IMAGENET)[0],
    }
}


sta_result_transformers = {
    'BERT': {
        'model_size': cal_mem_scale_trans(BERT_MEM)[1],
        'scale_mem_bs': cal_mem_scale_trans(BERT_MEM)[0],
    },
    'ROBERTA': {
        'model_size': cal_mem_scale(RESNET18_MEM_CIFAR)[1],
        'scale_mem_bs': cal_mem_scale(RESNET18_MEM_CIFAR)[0],
    },
}

# print(sta_result_transformers)
