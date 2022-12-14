from __future__ import print_function
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.utils.data.distributed
import tensorboardX
import torch.distributed as dist

from utils import init_distributed_mode
import time
from loader_ds import create_data_loader
from tqdm import tqdm

# from DynamicBatching import DyamicBatchingModule, MTrace
from qsync.LpTorch import QModule, QProfiler
from qsync.LpTorch.layers import convert_layers
from qsync.LpTorch import allreduce_hook
from qsync.LpTorch import config 
from qsync.Utils import get_capability
from torch.cuda.amp import GradScaler


profile_folder = os.path.join(__file__, os.pardir, os.pardir, "profile_result")

# run with bpslaunch python3 /qsync_niti_based/benchmark_classification/CIFAR_BENCH/main_bps.py
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.0125, metavar='LR',
                    help='learning rate (default: 0.0125)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')
parser.add_argument('--checkpoint-format', default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during pushpull')
parser.add_argument('--inference-only', action='store_true', default=False,
                    help='enable inference')

parser.add_argument('--enable-db', action='store_true', default=False,
                    help='enable dynamic batching')
parser.add_argument('--enable-qsync', action='store_true', default=False,
                    help='enable qsync')
parser.add_argument('--enable-simu', action='store_true', default=False,
                    help='enable qsync but using simulation')
parser.add_argument('--simu-arch', type=int, help='the arch to simu')
parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

# db realted

parser.add_argument('-db','--dy-bs', type=int, default=None, metavar='N',
                    help='dynamic initialization batchsize')
parser.add_argument('--model-name', type=str, default='VGG16',
                    help='model name')
parser.add_argument('--ds-name', type=str, default='cifar10',
                    help='data set name')  


parser.add_argument('--train-dir', default=os.path.expanduser('~/imagenet/train'),
                    help='path to training data')
parser.add_argument('--val-dir', default=os.path.expanduser('~/imagenet/validation'),
                    help='path to validation data')

parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')


parser.add_argument('--profile-only', action='store_true', default=False,
                    help='disables CUDA training')
                    
parser.add_argument('--label-smoothing', default=0.0, type=float,
                        help='label smoothing (default: 0.0)',
                        dest='label_smoothing')

parser.add_argument('--async-method', type=str, help="use async SSP or ASP")
parser.add_argument('--ssp-threshold', type=int, default=5, help="ssp threshold")  
parser.add_argument('--enable-async', action='store_true', default=False, help='enable async training')

parser.add_argument('--trace-driven', action='store_true', default=False, help='enable mem trace driven experiment')
parser.add_argument('--mem-gap', type=int, default=100, help='number of iterations required for change the memory')
parser.add_argument('--fixed-mem', action='store_true', default=False, help='enable mem trace driven experiment')
parser.add_argument('--fixed-mem-rate', type=float, default=1.0, help='enable mem trace driven experiment')
parser.add_argument('--T-B', type=float, default=1.0, help='profiled TB of training unit')


parser.add_argument('--test-case', type=int, default=None, help='test_case')
parser.add_argument('--tf32', action='store_true', default=False, help='use tf32 for training')
parser.add_argument('--fp16', action='store_true', default=False, help='use fp16 for training')
parser.add_argument('--int8', action='store_true', default=False, help='use int8 for training')
# tricks
parser.add_argument('--clip', action='store_true', default=False, help='enable gradient clipping (to avoid nan)')
parser.add_argument('--autoscale', action='store_true', default=False, help='auto scale')
parser.add_argument('--channel_wise', action='store_true', default=False, help='channel-wise')

# runtime
parser.add_argument('--enable_runtime', action='store_true', default=False,
                    help='skip bn and bias')
parser.add_argument('--enable_adjustment', action='store_true', default=False,
                    help='skip bn and bias')
parser.add_argument('--enable_db', action='store_true', default=False,
                    help='skip bn and bias')
parser.add_argument('--turnon_period_collect', action='store_true', default=False,
                    help='turn on period collection')
parser.add_argument('--runtime_period', type=int, default=1, metavar='S',
                    help='the runtime ajustment period')
parser.add_argument('--mapper_path', type=str, default=None, help='mapped path')
parser.add_argument('--indicator_type', type=int, default=0, help='ind type')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

init_distributed_mode(args) # init

if args.channel_wise:
    config.act_quant = 'scale'
    config.kernel_quant = 'channel'
    

cudnn.benchmark = True
# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for try_epoch in range(args.epochs, 0, -1) :
    if os.path.exists(args.checkpoint_format.format(epoch=try_epoch)):
        resume_from_epoch = try_epoch
        break

# BytePS: print logs on the first worker.
verbose = 1 if hasattr(args, "gpu") and args.gpu == 0 else 0
# BytePS: write TensorBoard logs on first worker.
log_writer = tensorboardX.SummaryWriter(args.log_dir) if  hasattr(args, "rank") and args.rank == 0 else None
# change batchsize
if args.dy_bs is None:
    args.dy_bs = args.batch_size

# get dy bs dict
if args.distributed:
    comm_num = 2 # rank and dy_bs
    comm_list = [torch.zeros(comm_num, dtype=torch.int64).cuda() for _ in range(args.world_size)]
    comm_vector = torch.tensor([args.rank, args.dy_bs]).cuda()

    if args.world_size > 1:
        dist.all_gather(comm_list, comm_vector)

    bs_dynamic = {item[0].item():item[1].item() for item in comm_list}
    args.bs_dynamic = bs_dynamic
    del comm_list, comm_vector



train_loader, train_sampler, test_loader, test_sampler = create_data_loader(args) # for imagenet it is actually 

from models import *

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)

def get_model(model_name):
    model_name = model_name.upper()
    if model_name == 'VGG16':
        if args.ds_name == 'cifar10':
            return VGG('VGG16')
        return models.vgg16()
    if model_name == 'VGG16BN':
        return models.vgg16_bn()
    if model_name == 'VGG13BN':
        return models.vgg13_bn()
    if model_name == 'VGG19':
        # if args.ds_name == 'cifar10':
        #     return VGG('VGG19')
        return models.vgg19()
    if model_name == 'VGG19BN':
        # if args.ds_name == 'cifar10':
        #     return VGG('VGG19')
        return models.vgg19_bn()
    if model_name == 'RESNET18':
        if args.ds_name == 'cifar10':
            return ResNet18()
        return models.resnet18()
    
    if model_name == 'RESNET34':
        if args.ds_name == 'cifar10':
            return ResNet34()
        return models.resnet34()
    if model_name == 'RESNET50':
        if args.ds_name == 'cifar10':
            return ResNet50()
        return models.resnet50()
    
    if model_name == 'RESNET101':
        if args.ds_name == 'cifar10':
            return ResNet101()
        return models.resnet101()
    
    if model_name == 'RESNET152':
        if args.ds_name == 'cifar10':
            return ResNet101()
        return models.resnet152()


    if model_name == 'RESNEXT':
        return ResNeXt29_2x64d()
    
model_name = args.model_name 
print("Use model: ", model_name)
model = get_model(model_name)
criterion = nn.CrossEntropyLoss()
train_criterion = nn.CrossEntropyLoss(reduction='sum')
arch_num = str(get_capability())

if args.distributed:
    rank = args.rank
    local_rank = args.gpu
    _world_size = args.world_size
else:
    args.rank = rank = 0
    args.gpu = local_rank = 0
    args.world_size = _world_size = 1


if args.distributed:
    all_bs = torch.tensor([args.dy_bs]).cuda()
    dist.all_reduce(all_bs)
    # print(all_bs, args.batch_size, args.dy_bs)
    # the gradient should contributes same
    # ddp generate mean of local gradient, aggregate it with all ranks
    # to contribute the same, we have to 
    loss_scale_ratio = round(args.dy_bs / args.batch_size, 10) # avoid overflow 
    all_bs = all_bs.item()
    dist.barrier()
else:
    loss_scale_ratio = 1.0

if args.enable_simu:
    arch_num = args.simu_arch

if config.simu:
    model = QModule(model, enable_simu=True, simu_arch=75, enable_runtime=args.enable_runtime, rank=rank, local_rank=local_rank, _world_size=_world_size, \
     is_adjustment_node=args.enable_adjustment, runtime_period=args.runtime_period, turnon_period_collect=args.turnon_period_collect,\
     indicator_type=args.indicator_type)
else:
    model = QModule(model, enable_runtime=args.enable_runtime, rank=rank, local_rank=local_rank, _world_size=_world_size, \
    is_adjustment_node=args.enable_adjustment, runtime_period=args.runtime_period, turnon_period_collect=args.turnon_period_collect, \
    indicator_type=args.indicator_type)

# model.load_and_set_depth(model_name)
model.cuda()
# print(model)

profile_path = os.path.abspath(os.path.join(profile_folder, f"profile_data_{model_name.lower()}_{arch_num}_{config.indicator_type}"))
if args.profile_only:
    QProfiler(model, train_loader, train_criterion, rank=model.local_rank, profile_iter=10, available_bits=model.available_bits, profiled_data_path=profile_path)
    exit()

    

if not args.profile_only:
    if os.path.exists(profile_path + '.npz'):
        model.load_profile_file(profile_path + '.npz')


exp_name = 'normal'
if args.test_case is not None:
    test_case_num = args.test_case
    if test_case_num == 1:
        model.set_bits(32)
        exp_name = 'fp32_comm_t4'
    elif test_case_num == 2:
        model.set_bits(16)
        exp_name = 'fp16_comm_t4'
    elif test_case_num == 3:
        model.set_bits(8)
        exp_name = 'int8_comm_t4'

    elif test_case_num == 4:
        target_layer = model.model.layer1
        model.set_bits_in_layers(16, target_layer)
        target_layer = model.model.layer3
        model.set_bits_in_layers(16, target_layer)
        exp_name = 'fp16_layer2_comm_t4'

    elif test_case_num == 5:
        qconv_list = model.qconv_list
        for qconv in qconv_list:
            qconv.reset_bits(16)
        exp_name = 'fp16_conv_comm_t4'
    
    elif test_case_num == 6:
        qconv_list = model.qconv_list
        for qconv in qconv_list:
            qconv.reset_bits(8)

        exp_name = 'int8_conv_comm_t4'
    
    elif test_case_num == 7:
        if arch_num == 75 or arch_num == '75':
            model.set_bits(16)
        else:
            model.set_bits(32)
        exp_name = f'mixed_arch{arch_num}'

if args.tf32:
    qconv_list = model.qconv_list
    for qconv in qconv_list:
        qconv.reset_bits(19)

if args.fp16:
    qconv_list = model.qconv_list
    for qconv in qconv_list:
        qconv.reset_bits(16)
if args.int8:
    qconv_list = model.qconv_list
    for qconv in qconv_list:
        qconv.reset_bits(8)

# model.load_and_set_depth(model_name)
print("arg path", args.mapper_path)
if args.mapper_path is not None:
    model.set_mapper_result(args.mapper_path)
    print("Set new bit")


# BytePS: scale learning rate by the number of GPUs.
if args.ds_name == 'cifar10':
    optimizer = optim.SGD(model.parameters(), lr=args.lr * _world_size / 4,
                      momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.wd)

if args.ds_name == 'cifar10':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)


model_without_ddp = model
if args.distributed:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    # model.register_comm_hook(state=None, hook=allreduce_hook)
    model_without_ddp = model.module.model

if resume_from_epoch > 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath, map_location=torch.device(args.gpu))
    model_without_ddp.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    


start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)


if args.autoscale:
    scaler = GradScaler()

def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    # BytePS: set epoch to sampler for shuffling.
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    lr=optimizer.param_groups[0]["lr"]
    iters = 0
    
    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}m lr={}'.format(epoch, lr),
              disable=not verbose) as t:
        
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=50, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log/{model_name}@{exp_name}'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
        ) as prof:
            for batch_idx, (data_, target_) in enumerate(train_loader):
                if args.ds_name == 'imagenet':
                    adjust_learning_rate(epoch, batch_idx)
                # print(args.rank, len(data_))
                data, target = data_, target_
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                
                if not config.simu:
                    data = data.to(memory_format=torch.channels_last)
                # model_without_ddp.advance_q_weight_calculate()
                start.record()
                output = model(data)
                # if args.enable_qsync:
                    # model.get_last_layer_result()
                    # model.communicate_last_layer()
                
                loss = train_criterion(output, target)
                loss /= (all_bs / _world_size)
                train_loss.update(loss.item())
                if args.autoscale:
                    loss = scaler.scale(loss)

                # allocated = torch.cuda.memory_allocated(0)
                # reserved = torch.cuda.memory_reserved(0)
                # print("allocated: %.2f MB" % (allocated / 1024 / 1024), flush=True)
                # print("reserved:  %.2f MB" % (reserved / 1024 / 1024), flush=True)


                # import pdb; pdb.set_trace()
                loss.backward()
                end.record()
                torch.cuda.synchronize()
                Duration = start.elapsed_time(end)

                # torch.cuda.synchronize()
                # if args.rank == 0:
                #     print("Method Cost", time.time() - start_time)
                
                # optimizer.synchronize()
                if args.clip:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2) 
                # calculate weight in advance

                if args.autoscale:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                if args.trace_driven and not args.fixed_mem:
                    new_ratio = MTIns.step()
                    if new_ratio is not None:
                        if not args.enable_qsync:
                            model.init_mem(new_ratio) # adjust the batchsize / or quantization level
                # if batch_idx % args.log_interval == 0:
                    # BytePS: use train_sampler to determine the number of examples in
                    # this worker's partition.
                
                train_accuracy.update(accuracy(output, target).item())
                
                t.set_postfix({'loss': train_loss.avg,
                            'accuracy': 100. * train_accuracy.avg})
                t.update(1)

                if args.test_case is not None:
                    if batch_idx >= 55:
                        exit()
                prof.step()

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)



        
        

def metric_average(val, name):
    tensor = torch.tensor(val)
    if args.cuda:
        tensor = tensor.cuda()
    avg_tensor = bps.push_pull(tensor, name=name)
    return avg_tensor.item()


best_acc = 0
def test(epoch):
    global best_acc
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with tqdm(total=len(test_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.cuda(), targets.cuda()
                if not config.simu:
                    inputs = inputs.to(memory_format=torch.channels_last)
                output = model(inputs)
                # sum up batch loss
                test_loss = criterion(output, targets)
                # get the index of the max log-probability
                val_loss.update(test_loss.item())
                val_accuracy.update(accuracy(output, targets).item())
                t.set_postfix({'loss': val_loss.avg,
                               'accuracy': 100. * val_accuracy.avg})
                t.update(1)
    
    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)

    
    # BytePS: print output only on first rank.
    if verbose:
        # Save checkpoint.
        acc = 100. * val_accuracy.avg
        if acc > best_acc:
            save_checkpoint(epoch)
            best_acc = acc

def adjust_learning_rate(epoch, batch_idx):
    if epoch <= args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / args.world_size * (epoch * (args.world_size - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj

def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).float().mean()

def save_checkpoint(epoch):
    filepath = args.checkpoint_format.format(epoch=epoch + 1)
    state = {
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)

import math
# BytePS: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = 0
        self.n = 0

    def update(self, val):
        self.sum += val
        self.n += 1

        # if math.isnan(self.sum):
        #     self.sum = 1
        #     self.n = 1

    @property
    def avg(self):
        return self.sum / self.n
        
if args.inference_only:
    test(1)
    exit()
start_wall_time = time.time()
for epoch in range(resume_from_epoch, args.epochs + 1):
    if args.ds_name == 'cifar10':
        scheduler.step()
    train(epoch)
    test(epoch)
    wall_time = time.time() - start_wall_time
    print(f"epoch [{epoch}]: walltime-", wall_time)