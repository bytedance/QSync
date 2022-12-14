'''
    DataLoader for exchanging the result.
'''
from torchvision import datasets, transforms, models
import torchvision
import os, torch 

from DynamicBatching import DBDistributedSampler
# from DynamicBatching import DBDDPSampler, DBBatchSampler
def get_imagenet(args):
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_dataset = \
        datasets.ImageFolder(args.train_dir,
                            transform=transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(
    #         train_dataset)
    # else:
    #     train_sampler = None

    # rank and its bs

    if not hasattr(args, 'bs_dynamic') or args.bs_dynamic is None:
        bs_dynamic = {i:args.batch_size for i in range(args.world_size)}
    else:
        bs_dynamic = args.bs_dynamic
        
    if args.distributed:
        train_sampler = DBDistributedSampler(
            train_dataset, bs_dynamic=bs_dynamic, batch_size=args.batch_size)
    else:
        train_sampler = None 

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs_dynamic[args.rank], sampler=train_sampler, **kwargs)


    val_dataset = \
        datasets.ImageFolder(args.val_dir,
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                            ]))
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset) if args.distributed else None
    # val_sampler = DBDistributedSampler(
    #     val_dataset) if args.distributed else None

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                         sampler=val_sampler, **kwargs)


    return train_loader, train_sampler, val_loader, val_sampler

def get_cifar10(args):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=f'{os.environ["HOME"]}/data/data-{args.rank}', train=True, download=True, transform=transform_train)

    test_dataset = torchvision.datasets.CIFAR10(
        root=f'{os.environ["HOME"]}/data/data-{args.rank}', train=False, download=True, transform=transform_test)


    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    # BytePS: use DistributedSampler to partition the training data.

    if not hasattr(args, 'bs_dynamic') or args.bs_dynamic is None:
        bs_dynamic = {i:args.batch_size for i in range(args.world_size)}
    else:
        bs_dynamic = args.bs_dynamic
        
    if args.distributed:
        train_sampler = DBDistributedSampler(
            train_dataset, bs_dynamic=bs_dynamic, batch_size=args.batch_size)
    else:
        train_sampler = None 

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     train_dataset, num_replicas=args.world_size, rank=args.rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs_dynamic[args.rank], sampler=train_sampler, **kwargs)

    # BytePS: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=args.world_size, rank=args.rank)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                          sampler=test_sampler, **kwargs)

    return train_loader, train_sampler, test_loader, test_sampler


def create_data_loader(args):
    if args.ds_name == 'imagenet':
        return get_imagenet(args)
    elif args.ds_name == 'cifar10':
        return get_cifar10(args)