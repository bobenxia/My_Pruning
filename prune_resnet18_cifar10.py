import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprof
from torchinfo import summary
from torchvision import transforms
from torchvision.datasets import CIFAR10

import model.cifar.resnet as resnet
import torch_pruning as tp
from model.cifar.resnet import ResNet18
from tools.inference_time import (count_params, get_cpu_gpu_time_in_inference,
                                  measure_inference_time)

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=30)
parser.add_argument('--round', type=int, default=1)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
local_rank = args.local_rank


def get_dataloader():
    train_loader = torch.utils.data.DataLoader(CIFAR10('/data/xiazheng/',
                                                       train=True,
                                                       transform=transforms.Compose([
                                                           transforms.RandomCrop(32, padding=4),
                                                           transforms.RandomHorizontalFlip(),
                                                           transforms.ToTensor(),
                                                       ]),
                                                       download=True),
                                               batch_size=args.batch_size,
                                               num_workers=2)
    test_loader = torch.utils.data.DataLoader(CIFAR10('/data/xiazheng/',
                                                      train=False,
                                                      transform=transforms.Compose([
                                                          transforms.ToTensor(),
                                                      ]),
                                                      download=True),
                                              batch_size=args.batch_size,
                                              num_workers=2)
    return train_loader, test_loader


def eval(model, test_loader):
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred == target).sum()
            total += len(target)
    return correct / total


def train_model(model, train_loader, test_loader):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    model.to(local_rank)
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    best_acc = -1

    import nvidia_dlprof_pytorch_nvtx as nvtx
    nvtx.init(enable_function_stack=True)
    with torch.autograd.profiler.emit_nvtx():
        for epoch in range(args.total_epochs):
            model.train()
            for i, (img, target) in enumerate(train_loader):
                img, target = img.to(local_rank), target.to(local_rank)
                optimizer.zero_grad()
                out = model(img)
                loss = loss_func(out, target)
                loss.backward()
                optimizer.step()
                if i % 10 == 0 and args.verbose:
                    print("Epoch %d/%d, iter %d/%d, loss=%.4f" %
                          (epoch, args.total_epochs, i, len(train_loader), loss.item()))
            model.eval()
            acc = eval(model, test_loader)
            print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))
            if best_acc < acc:
                torch.save(model.module, 'ResNet18-round%d.pth' % (args.round))
                best_acc = acc
            scheduler.step()
        print("Best Acc=%.4f" % (best_acc))


def prune_model(model):
    model.cpu()
    DG = tp.DependencyGraph().build_dependency(model, torch.randn(1, 3, 32, 32))

    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        # pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        print(amount, len(pruning_index))
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet18(num_classes=10)
        # ckpt = 'ResNet18-round%d.pth'%(args.round)
        # print("Load model from %s"%( ckpt ))
        # model = torch.load( ckpt )
        train_model(model, train_loader, test_loader)
    elif args.mode == 'prune':
        previous_ckpt = 'save/train_and_prune/ResNet18-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        prune_model(model)
        # print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        torch.save(model, 'save/train_and_prune/ResNet18-round%d.pth' % (args.round))
        # train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = 'save/train_and_prune/ResNet18-round%d.pth' % (args.round)
        # ckpt = 'save/train_and_prune/ResNet18-round0-cp.pth'
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))

        device = torch.device('cuda')
        fake_input = torch.randn(1, 3, 32, 32).to(device)
        model = model.to(device)

        cpu_time, gpu_time = get_cpu_gpu_time_in_inference(model, fake_input)
        print("before pruning: inference time=%f s, parameters=%.1fM" %
              (inference_time_before_pruning, count_params(model) / 1e6))
        # inference_time_before_pruning = measure_inference_time(model, fake_input, repeat)
        # with torchprof.Profile(model, use_cuda=True) as prof:
        #     _ = model(fake_input)
        # print(prof.display(show_events=False))
        # summary(model, (1,3,32,32))
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False,use_cpu=False, use_kineto=True) as prof:
        #     _ = model(fake_input)
        # print(prof.table())
        # print("before pruning: inference time=%f s, parameters=%.1fM"%(inference_time_before_pruning, count_params(model)/1e6))


if __name__ == '__main__':
    main()
