from typing import Any, Callable
import torch
import torch.distributed as dist


# collection hooks

def backward_fp32_hook(module, grad_output, grad_input):
    # print( module.weight.shape)
        module.weight.grad = module.weight.grad.float()

def backward_collection_hook(module, grad_output, grad_input):
    with torch.no_grad():
        # print(grad_output, module.name)
        grad_output_0 = grad_output[0].detach() if grad_output[0] is not None else 1
        # previous studies has shown that scaling and running avg has no apparent difference. we adopted here
        if type(grad_output_0) is int:
            grad_v_norm2 = 1
        else:
            grad_v_norm2 = torch.norm(grad_output_0).item()
        module.grad_v_norm2 = grad_v_norm2
        # if config.enable_period_collect:
        #     if hasattr(module, 'nabla_cnt'):
        #         # module.nabla_cnt += 1
        #         module.nabla_v = calculate_running_avg(module.nabla_v, module.nabla_cnt, grad_output_0)
        #     else:
        #         # module.nabla_cnt = 1
    #         module.nabla_v = grad_output_0

def forward_collection_hook(module, input, output):
    input_0 = input[0].detach().cpu().numpy()
    weight_0 = module.weight.detach().cpu().numpy()
    if config.enable_period_collect:
        if hasattr(module, 'fwd_cnt'):
            module.fwd_cnt += 1
            module.v = calculate_running_avg(module.v, module.fwd_cnt, input_0)
            module.x = calculate_running_avg(module.x, module.fwd_cnt, weight_0)
    else:
        module.fwd_cnt = 1
        module.v = input_0
        module.x = weight_0

def register_bwd_collection_hook(mod):
    hook_bwd = mod.register_full_backward_hook(backward_collection_hook)
    return hook_bwd

def register_collection_hooks(mod):
    hook_fwd = mod.register_forward_hook(forward_collection_hook)
    hook_bwd = register_bwd_collection_hook(mod)
    return hook_fwd, hook_bwd
        


# comm hooks
# reference
def fp16_compress_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision floating-point format (``torch.float16``)
    and then divides it by the process group size.
    It allreduces those ``float16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).
    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, fp16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.float16).div_(world_size)

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)



def fp16_fp32agg_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
        convert fp16 / fp32 -> fp32 for aggregation then compressed it to fp16 / fp32
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()

    compressed_tensor = bucket.buffer().to(torch.float32).div_(world_size)
    # since the gradient type can be different in our case. use gradients
    # gradients = bucket.gradients()

    fut = dist.all_reduce(
        compressed_tensor, group=group_to_use, async_op=True
    ).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        # Decompress in place to reduce the peak memory.
        # See: https://github.com/pytorch/pytorch/issues/45968
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor

    return fut.then(decompress)



def _allreduce_fut(
    process_group: dist.ProcessGroup, tensor: torch.Tensor
) -> torch.futures.Future[torch.Tensor]:
    "Averages the input gradient tensor by allreduce and returns a future."
    group_to_use = process_group if process_group is not None else dist.group.WORLD

    # Apply the division first to avoid overflow, especially for FP16.
    tensor.div_(group_to_use.size())

    return (
        dist.all_reduce(tensor, group=group_to_use, async_op=True)
        .get_future()
        .then(lambda fut: fut.value()[0])
    )


def allreduce_hook(
    process_group: dist.ProcessGroup, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """
    This DDP communication hook just calls ``allreduce`` using ``GradBucket``
    tensors. Once gradient tensors are aggregated across all workers, its ``then``
    callback takes the mean and returns the result. If user registers this hook,
    DDP results is expected to be same as the case where no hook was registered.
    Hence, this won't change behavior of DDP and user can use this as a reference
    or modify this hook to log useful information or any other purposes while
    unaffecting DDP behavior.

    Example::
        >>> ddp_model.register_comm_hook(process_group, allreduce_hook)
    """
    print(bucket.buffer().shape)
    return _allreduce_fut(process_group, bucket.buffer())