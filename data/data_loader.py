import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
from .distributed_sampler import OrderedDistributedSampler
from .dataset import Dataset

def create_dataset_loader(dataset, dataset_dir, train_portion, batch_size, workers, distributed=False):
    dataset = dataset.lower()
    assert dataset in ['cifar100','imagenet']
    if dataset == 'cifar100':
        return create_cifar100_loader(dataset_dir, train_portion, batch_size, workers)
    elif dataset == 'imagenet':
        return create_imagenet_loader(dataset_dir, train_portion, batch_size, workers, distributed=distributed)

def create_cifar100_loader(dataset_dir, train_portion, batch_size, workers):
    train_transform, valid_transform = _data_transforms_cifar100()
    train_data = dset.CIFAR100(root=dataset_dir, train=True, download=False, transform=train_transform)
    test_data = dset.CIFAR100(root=dataset_dir, train=False, download=False, transform=valid_transform)

    num_test = len(test_data)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.round(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True,
        num_workers=workers
    )
    valid_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True,
        num_workers=workers
    )
    test_queue = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:num_test]),
        pin_memory=True,
        num_workers=workers
    )
    return train_queue, valid_queue, test_queue

def _data_transforms_cifar100():
    CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
    CIFAR_STD =  [0.2675, 0.2565, 0.2761]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return train_transform, valid_transform

def create_imagenet_loader(dataset_dir, train_portion, batch_size, workers, distributed):
    train_dir = os.path.join(dataset_dir, 'train')
    valid_dir = os.path.join(dataset_dir,'val')
    train_data = Dataset(train_dir)
    valid_data = Dataset(valid_dir)

    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    train_data.transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    valid_data.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    sampler_train = None
    sampler_test = None
    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_data)
        sampler_test = OrderedDistributedSampler(valid_data)
    train_queue = torch.utils.data.DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = sampler_train is None,
        num_workers = workers,
        sampler = sampler_train,
        collate_fn = torch.utils.data.dataloader.default_collate,
        drop_last = True
    )
    valid_queue = torch.utils.data.DataLoader(
        valid_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = workers,
        sampler = sampler_test,
        collate_fn = torch.utils.data.dataloader.default_collate,
        drop_last = False
    )
    return train_queue, valid_queue, valid_queue

