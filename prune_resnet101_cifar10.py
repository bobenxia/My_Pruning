import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchprof
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

import model.cifar.resnet as resnet
import torch_pruning as tp
from model.cifar.resnet import ResNet101
from tools.inference_time import count_params, measure_inference_time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    default='test',
                    choices=['train', 'prune', 'test', 'finetune'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=30)
parser.add_argument('--step_size', type=int, default=30)
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


def train_model(model, train_loader, test_loader, summary_writer, name=None):
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
    save_name = 'save/train_and_prune/ResNet101-round%d.pth' % (args.round) if not name else \
                'save/train_and_prune/ResNet101-round%d-%s.pth' % (
                    args.round, name)

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
            torch.save(model.module, save_name)
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
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]
    blk_id = 0
    for m in model.modules():
        if isinstance(m, resnet.Bottleneck):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            blk_id += 1

    torch.save(model, 'save/train_and_prune/ResNet101-round%d.pth' % (args.round))
    return model


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet101(num_classes=10)
        train_model(model,
                    train_loader,
                    test_loader,
                    summary_writer="./runs/prune_resnet101_cifar10.log")
    elif args.mode == 'prune':
        previous_ckpt = 'save/train_and_prune/ResNet101-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters before prune: %.1fM" % (params / 1e6))
        prune_model(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters after prune: %.1fM" % (params / 1e6))
    elif args.mode == 'finetune':
        previous_ckpt = 'save/train_and_prune/ResNet101-round%d.pth' % (args.round)
        print("Finetune round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt)
        train_model(model,
                    train_loader,
                    test_loader,
                    summary_writer="./runs/prune_resnet101_cifar10_after_prune.log",
                    name='finetune')
    elif args.mode == 'test':
        ckpt = 'save/train_and_prune/ResNet101-round%d.pth' % (args.round)
        # ckpt = 'save/train_and_prune/ResNet101-round1-finetune.pth'
        print("Load model from %s" % (ckpt))
        model = torch.load(ckpt)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM" % (params / 1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n" % (acc))

        repeat = 1000
        device = torch.device('cuda')
        fake_input = torch.randn(1, 3, 32, 32).to(device)
        model = model.to(device)

        # method 1
        inference_time_before_pruning = measure_inference_time(model, fake_input, repeat)
        # print("inference time=%f s, parameters=%.1fM" %
        #       (inference_time_before_pruning, count_params(model)/1e6))

        # method 2
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=True, profile_memory=False,use_cpu=False, use_kineto=True) as prof:
        # with torch.autograd.profiler.profile(enabled=True, use_cuda=True, record_shapes=False, profile_memory=False) as prof:
        #     _ = model(fake_input)
        # print(prof.table())

        # method 3
        # with torchprof.Profile(model, use_cuda=True) as prof:
        #     _ = model(fake_input)
        # print(prof.display(show_events=True))
        # print(prof.display(show_events=True))
        # trace, event_lists_dict = prof.raw()
        # print(event_lists_dict[trace[1].path][0])
        # conv1 = event_lists_dict[trace[1].path][0]
        # print(conv1[0])
        # print(conv1[0].cuda_time)

        # method 4
        # from torch.profiler import profile, record_function, ProfilerActivity
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, with_stack=True) as prof:
        #     with record_function("model_inference"):
        #         _ = model(fake_input)
        # key_av = prof.key_averages()
        # for i in range(len(key_av)):
        #     if 'model_inference' in key_av[i].key:
        #         key = key_av[i].key
        #         print('cuda time total:', key_av[i].cuda_time_total_str)
        #         print('cpu time tital:', key_av[i].cpu_time_total_str)
        #         print('cuda memory usage:', key_av[i].cuda_memory_usage)
        # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        # prof.export_chrome_trace("trace.json")
        # prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")


if __name__ == '__main__':
    main()
