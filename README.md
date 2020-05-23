# Network Adjustment

![pytorch](https://img.shields.io/badge/pytorch-v1.0.0-green.svg?style=plastic)

## Introduction
This repository contains a pytorch implementation for the NetworkAdjustment algorithm. NetworkAdjustmetn can 
imporve the network performance by adjusting the channel numbers without introducing extra FLOPs.

## Usage
### channel search
Single GPU on CIFAR-100:
```bash
channel_search_distributed.py --dataset=cifar100 --dataset_dir=$DATA_DIR$ --gpu=0 --batch_size=128 --learning_rate=0.15 --arch=resnet_cifar --depth=20 --drop_rate=0.05 --base_drop_rate=0.05
```
Multi-GPU ( *e.g.* 8 GPU) on ImageNet:
```bash
./distributed_search.sh 8 --dataset=ImageNet --dataset_dir=$DATA_DIR$ --batch_size=64 --learning_rate=0.2 --arch=resnet_imagenet --classes=1000 --drop_rate=0.05 --base_drop_rate=0.05 --depth=18 --weight_decay=1e-5
```
## Results
#### Results on CIFAR-100:
![results on ImageNet](tables_and_imgs/table_cifar.png "ImageNet")

#### Results on ImageNet:
![results on ImageNet](tables_and_imgs/table_imagenet.png "ImageNet")

#### Searched ResNet18 0.8x on ImageNet:
![Search ResNet18](tables_and_imgs/resnet18_searched.png "Search ResNet18")

## Citation
If you use NetworkAdjustment in your research, please cite the paper:
```
@article{Chen2020NetworkAdjustment,
  title={Network Adjustment: Channel Search Guided by FLOPs Utilization Ratio},
  author={Chen, Zhengsu and Niu, Jianwei and Xie, Lingxi and Liu, Xuefeng and Wei, Longhui and Tian, Qi},
  journal={arXiv preprint arXiv:2004.02767},
  year={2020}
}
```
