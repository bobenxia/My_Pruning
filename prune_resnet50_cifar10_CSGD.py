#  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 prune_resnet101_cifar10.py --mode finetune
import argparse
import os
import sys
import time
from email.policy import default

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from django.template import Engine
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10

import model.cifar.resnet as resnet
import torch_pruning as tp
from model.cifar.resnet import ResNet50
from tools.print_model_info import get_model_infor_and_print
from tools.write_excel import read_excel_and_write
from utils.model_utils import *

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
parser.add_argument('--out_dir', type=str, default='save/')

args = parser.parse_args()
local_rank = args.local_rank
block_prune_probs = [args.pruned_per] * 16


def get_dataloader():
    train_loader = DataLoader(CIFAR10('/Tos/cifar_10',
                                      train=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                      ]),
                                      download=True),
                              batch_size=args.batch_size,
                              num_workers=2)
    test_loader = DataLoader(CIFAR10('/Tos/cifar_10',
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



def update_net_params(net, param_name_to_merge_matrix, param_name_to_decay_matrix):
    # C-SGD works here
    for name, param in net.named_parameters():
        name = name.replace('module.', '')
        if name in param_name_to_merge_matrix:
            p_dim = param.dim()
            p_size = param.size()
            if p_dim == 4:
                param_mat = param.reshape(p_size[0], -1)
                g_mat = param.grad.reshape(p_size[0], -1)
            elif p_dim == 1:
                param_mat = param.reshape(p_size[0], 1)
                g_mat = param.grad.reshape(p_size[0], 1)
            else:
                assert p_dim == 2
                param_mat = param
                g_mat = param.grad
            # 上面是获取当前的梯度，reshape 成 g_mat
            # 下面是将 g_mat 按照文章中的公式，进行矩阵相乘和相加。
            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(param_mat)
            # 将计算的结果更新到参数梯度中。
            param.grad.copy_(csgd_gradient.reshape(p_size))


def train_with_csgd(model, train_loader, test_loader, summary_writer, clusters_save_path):
    # deps, target_deps, schedule 是需要手动设置的
    schedule = 0.75
    deps = RESNET50_ORIGIN_DEPS_FLATTENED # resnet50 的 通道数量
    target_deps = generate_itr_to_target_deps_by_schedule_vector(schedule, RESNET50_ORIGIN_DEPS_FLATTENED, RESNET50_INTERNAL_KERNEL_IDXES)
    # pacesetter_dict 也是需要手动设置的
    pacesetter_dict ={4: 4, 3: 4, 7: 4, 10: 4, 14: 14, 13: 14, 17: 14, 20: 14, 23: 14, 27: 27, 26: 27, 30: 27, 33: 27, 36: 27, 39: 27, 42: 27, 46: 46, 45: 46, 49: 46, 52: 46}

    with Engine(local_rank=local_rank) as engine:
        # ------------------------ parepare optimizer, scheduler, criterion -------
        optimizer = torch.optim.SGD(model.parameters(), lr=0.04, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        loss_func = nn.CrossEntropyLoss().to(local_rank)
        engine.register_state(scheduler=scheduler, model=model, optimizer=optimizer)
        # --------------------------------- done -------------------------------


        # DDP：DDP backend初始化
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
        model.to(local_rank)
        # DDP: 构造DDP model
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        best_acc = -1

        # summarywriter
        writer = SummaryWriter(summary_writer)

        #  ========================== prepare the clusters and matrices for  C-SGD =======================
        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list()
        if os.path.exists(clusters_save_path):
            layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()
        else:
            if local_rank == 0:
                # 获取聚类的 idx 结果
                # # 返回的是一个字典，每个 key 是层id, 对应的 value 的值是一个长度等于当前层长度的聚类结果。[[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
                layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                                target_deps=target_deps,
                                                                pacesetter_dict=pacesetter_dict)
                # pacesetter_dict 是残差结构的连接层之间的关系。
                if pacesetter_dict is not None:
                    for follower_idx, pacesetter_idx in pacesetter_dict.items():
                        # 这里将残差的最后一层的剪枝方案直接等同于直连的剪枝方案
                        if pacesetter_idx in layer_idx_to_clusters:
                            layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]
                # 保存聚类的 idx 结果
                np.save(clusters_save_path, layer_idx_to_clusters)
            else:
                while not os.path.exists(clusters_save_path):
                    time.sleep(10)
                    print('sleep, waiting for process 0 to calculate clusters')
                layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()

        # 根据 聚类的通道的结果，生成 matrix，方便计算
        param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=deps,
                                                                      layer_idx_to_clusters=layer_idx_to_clusters,
                                                                      kernel_namedvalue_list=kernel_namedvalue_list)
        # 这块的功能似乎是要添加每层对于的 bias\gamma\beta 进入这个 param_name_to_merge_matrix
        add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix)
        # core code 聚类结果的梯度计算，作为新的 weight decay
        param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(deps=deps,
                                                                               layer_idx_to_clusters=layer_idx_to_clusters,
                                                                               kernel_namedvalue_list=kernel_namedvalue_list,
                                                                               weight_decay=1e-4,
                                                                               weight_decay_bias=0,
                                                                               centri_strength=0.06)

        for epoch in range(args.total_epochs):
            model.train()
            for i, (img, target) in enumerate(train_loader):
                img, target = img.to(local_rank), target.to(local_rank)
                optimizer.zero_grad()
                out = model(img)
                loss = loss_func(out, target)
                loss.backward()

                # update networl params based on the CSGD
                update_net_params(model, param_name_to_merge_matrix, param_name_to_decay_matrix)

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


def main():
    train_loader, test_loader = get_dataloader()
    if args.mode == 'train':
        args.round = 0
        model = ResNet50(num_classes=10)
        train_model(model, train_loader, test_loader, summary_writer="./runs/prune_resnet50_cifar10.log")
    elif args.mode == 'train_with-csgd':
        args.round = 0
        model = ResNet50(num_classes=10)
        train_with_csgd(model, train_loader, test_loader, , summary_writer="./runs/prune_resnet50_cifar10_CSGD.log"
                        , clusters_save_path="./save/train_and_prune")
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
