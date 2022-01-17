import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

import model.cifar.resnet as resnet
import torch_pruning as tp
from model.cifar.resnet import ResNet50
from tools.print_model_info import get_model_infor_and_print
from tools.write_excel import read_excel_and_write

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test', 'finetune'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=30)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--pruned_per', type=float, default=0.125)
parser.add_argument("--local_rank", default=-1, type=int)

args = parser.parse_args()
local_rank = args.local_rank
block_prune_probs = [args.pruned_per] * 16


def get_dataloader():
    train_loader = DataLoader(CIFAR10('/data/xiazheng/',
                                      train=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                      ]),
                                      download=True),
                              batch_size=args.batch_size,
                              num_workers=2)
    test_loader = DataLoader(CIFAR10('/data/xiazheng/',
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


def train_model(model, train_loader, test_loader, summary_writer):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    model.to(local_rank)
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    best_acc = -1

    # summarywriter
    writer = SummaryWriter(summary_writer)

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
            torch.save(model.module, 'save/train_and_prune/ResNet50-round%d.pth' % (args.round))
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
        strategy = tp.strategy.GSGDStrategy()
        pruning_index = strategy(conv.weight, amount=amount)
        print(amount, len(pruning_index))
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.Bottleneck):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet50(num_classes=10)
        train_model(model, train_loader, test_loader, summary_writer="./runs/prune_resnet50_cifar10.log")
    elif args.mode == 'prune':
        previous_ckpt = 'save/train_and_prune/ResNet50-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        prune_model(model)
        torch.save(model, 'save/train_and_prune/ResNet50-round%d.pth' % (args.round))
    elif args.mode == 'finetune':
        previous_ckpt = 'save/train_and_prune/ResNet50-round%d.pth' % (args.round)
        print("Finetune round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        train_model(model, train_loader, test_loader, summary_writer="./runs/prune_resnet50_cifar10_after_prune.log")
    elif args.mode == 'test':
        ckpt = 'save/train_and_prune/ResNet50-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))

        fake_input = torch.randn(1, 3, 32, 32)
        # need load model to cpu, avoid computing model GPU memory errors
        model = torch.load(ckpt, map_location=lambda storage, loc: storage)
        model_infor = get_model_infor_and_print(model, fake_input, 0)

        model_infor['Model'] = 'resnet-50'
        model_infor['Top1-acc(%)'] = eval(model, test_loader)
        model_infor['Top5-acc(%)'] = ' '
        model_infor['If_base'] = 'False'
        model_infor['Strategy'] = f'CSGD+{block_prune_probs}' if model_infor['If_base'] == 'False' else ' '
        print(model_infor)

        excel_path = "model_data.xlsx"
        read_excel_and_write(excel_path, model_infor)


if __name__ == '__main__':
    main()
