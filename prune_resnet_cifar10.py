import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

import model.cifar.resnet as resnet
import torch_pruning as tp
from config.default_config import _C as config
from model.cifar.resnet import ResNet34

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=20)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)
parser.add_argument("--local_rank", default=-1, type=int)


def get_dataloader(distributed):
    train_dataset = CIFAR10('/data/xiazheng/',
                            train=True,
                            transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ]),
                            download=True)
    test_dataset = CIFAR10('/data/xiazheng/',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                           ]),
                           download=True)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        drop_last=True)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.DATALOADER.BATCH_SIZE,
                                               sampler=train_sampler,
                                               num_workers=config.DATALOADER.NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.DATALOADER.BATCH_SIZE,
                                              sampler=test_sampler,
                                              num_workers=config.DATALOADER.NUM_WORKERS)
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


def train_model(model, train_loader, test_loader, args):
    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = len(gpus) > 1
    device = torch.device(f'cuda:{args.local_rank}')
    # DDP：DDP backend初始化
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl')

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    loss_func = nn.CrossEntropyLoss().to(device)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    else:
        model.to(device)

    best_acc = -1

    # summarywriter
    writer = SummaryWriter("./runs/prune_resnet34_cifar10.log")

    for epoch in range(args.total_epochs):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(device), target.to(device)
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
            torch.save(model.module, 'ResNet34-round%d.pth' % (args.round))
            best_acc = acc
        scheduler.step()
        writer.add_scalar('eval/acc', acc, epoch)
        writer.add_scalar('train/loss', loss, epoch)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
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
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    block_prune_probs = [
        0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3
    ]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1
    return model


def main():
    args = parser.parse_args()
    local_rank = args.local_rank
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet34(num_classes=10)
        train_model(model, train_loader, test_loader, args)
    elif args.mode == 'prune':
        previous_ckpt = 'ResNet34-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        # train_model(model, train_loader, test_loader)
    elif args.mode == 'test':
        ckpt = 'ResNet34-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))


if __name__ == '__main__':
    main()