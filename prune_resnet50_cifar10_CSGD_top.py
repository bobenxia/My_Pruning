#  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 prune_resnet101_cifar10.py --mode finetune
import argparse
import os
import sys
import time
from datetime import datetime
from email.policy import default
from functools import total_ordering

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
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
from utils.misc import copy_file, copy_files
from utils.misc import load_hdf5
from utils.cluster_params import generate_itr_for_model_follow_global_cluster

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    # required=True,
                    default='test',
                    choices=['train', 'prune', 'test', 'finetune', 'train_with_csgd'])
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=600)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--pruned_per', type=float, default=0.125)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--scheduler_', type=str, default='CAWR')
parser.add_argument('--lr_', type=float, default=0.04)

args = parser.parse_args()
local_rank = args.local_rank
block_prune_probs = [args.pruned_per] * 16
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
TENSORBOARD_LOG_DIR = os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/")



def get_dataloader():
    train_loader = DataLoader(CIFAR10('/tos/cifar_10',
                                      train=True,
                                      transform=transforms.Compose([
                                          transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                      ]),
                                      download=True),
                              batch_size=args.batch_size,
                              num_workers=2)
    test_loader = DataLoader(CIFAR10('/tos/cifar_10',
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


def train_model(model, model_name, train_loader, test_loader,model_save_path):
    # DDP：DDP backend初始化
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_, momentum=0.9, weight_decay=1e-4)
    if args.scheduler_ == "ELR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler_ == "CAWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=args.total_epochs//3+1, T_mult=2)
    else:
        raise ValueError("scheduler type error!")
    loss_func = nn.CrossEntropyLoss().to(local_rank)

    hdf5_file = f'/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.0003_0.25_600epoch/prune_mode.hdf5'
    print(hdf5_file)
    load_hdf5(model, hdf5_file)

    model.to(local_rank)

    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    best_acc = -1
    if local_rank == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR + TIMESTAMP)
    else:
        writer = None

    start_epoch, end_epoch = 0, args.total_epochs
    best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name, optimizer, scheduler, loss_func, best_acc, writer, start_epoch, end_epoch)

    if local_rank == 0:
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
        # name = name.replace('module.', '')  
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
            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(
                param_mat)
            # 将计算的结果更新到参数梯度中。
            param.grad.copy_(csgd_gradient.reshape(p_size))


def train_with_csgd(model, model_name, train_loader, test_loader, is_resume, model_save_path):
    # -------------------- 根据模型选择的一些超参 -------------------
    deps = RESNET50_ORIGIN_DEPS_FLATTENED  # resnet50 的 通道数量
    target_deps_for_kernel_matrix = generate_itr_to_target_deps_by_schedule_vector(1, RESNET50_ORIGIN_DEPS_FLATTENED, RESNET50_INTERNAL_KERNEL_IDXES)
    target_deps_for_decay_matrix = generate_itr_to_target_deps_by_schedule_vector(0, RESNET50_ORIGIN_DEPS_FLATTENED, RESNET50_INTERNAL_KERNEL_IDXES)
    pacesetter_dict = {
        4: 4,3: 4,7: 4,10: 4,14: 14,
        13: 14,17: 14,20: 14,23: 14,27: 27,26: 27,30: 27,33: 27,36: 27,
        39: 27,42: 27,46: 46,45: 46,49: 46,52: 46
    }
    # --------------------------- done ----------------------------

    # ------------ parepare optimizer, scheduler, criterion -------
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=args.total_epochs//3+1, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0.004, T_0=args.total_epochs+1, T_mult=2)
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    centri_strength=0.003
    # --------------------------- done ------------------------------

    # ------------------- DDP：DDP backend初始化 --------------------
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端
    model.to(local_rank)
    # DDP: 构造DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # -------------------------- done ------------------------------

    best_acc = -1

    # summarywriter
    if local_rank == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR + TIMESTAMP)
    else:
        writer = None

    # ------------------- 预训练 40 epoch,  然后进行全局聚类 ----------------
    start_epoch, end_epoch = 0, 40
    best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name+'-train', 
    optimizer, scheduler, loss_func, best_acc, writer, start_epoch, end_epoch)
    # ----------------------------------- done -----------------------------------

    # ------------------- 全局聚类 以及 后面的局部聚类 ----------------
    with ModelUtils(local_rank=local_rank) as engine:
        engine.setup_log(name='train', log_dir=model_save_path, file_name='ResNet50-CSGD-log.txt')
        engine.register_state(scheduler=scheduler, model=model, optimizer=optimizer)

        # ---------------------------  全局聚类    ------------------------------
        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list()

        #--------------------------- 1、生成 kernel martrix -----------------------------------
        clusters_save_path_for_kernel_matrix = os.path.join(model_save_path, 'clusters_for_kernel_matrix.npy')
        # clusters_save_path_for_kernel_matrix = 'save/train_and_prune/2022-02-08T17-37-47/clusters.npy'
        if os.path.exists(clusters_save_path_for_kernel_matrix):
            layer_idx_to_clusters_for_kernel_matrix = np.load(clusters_save_path_for_kernel_matrix, allow_pickle=True).item()
        else:
            if local_rank == 0:
                # 获取聚类的 idx 结果
                # # 返回的是一个字典，每个 key 是层id, 对应的 value 的值是一个长度等于当前层长度的聚类结果。[[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
                # 生成 layer_idx_to_clusters_for_kernel_matrix 为了计算 kernel matrix
                layer_idx_to_clusters_for_kernel_matrix = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                                  target_deps=target_deps_for_kernel_matrix,
                                                                  pacesetter_dict=pacesetter_dict)
                # pacesetter_dict 是残差结构的连接层之间的关系。
                if pacesetter_dict is not None:
                    for follower_idx, pacesetter_idx in pacesetter_dict.items():
                        # 这里将残差的最后一层的剪枝方案直接等同于直连的剪枝方案
                        if pacesetter_idx in layer_idx_to_clusters_for_kernel_matrix:
                            layer_idx_to_clusters_for_kernel_matrix[follower_idx] = layer_idx_to_clusters_for_kernel_matrix[pacesetter_idx]
                # 保存聚类的 idx 结果
                np.save(clusters_save_path_for_kernel_matrix, layer_idx_to_clusters_for_kernel_matrix)
            else:
                while not os.path.exists(clusters_save_path_for_kernel_matrix):
                    time.sleep(10)
                    print('sleep, waiting for process 0 to calculate clusters')
                layer_idx_to_clusters_for_kernel_matrix = np.load(clusters_save_path_for_kernel_matrix, allow_pickle=True).item()
        # ----------------------------------- done -----------------------------------

        #--------------------------- 2、生成 kernel martrix -----------------------------------
        clusters_save_path_for_decay_matrix = os.path.join(model_save_path, 'clusters_for_decay_matrix.npy')
        # clusters_save_path_for_decay_matrix = 'save/train_and_prune/2022-02-08T17-37-47/clusters.npy'
        if os.path.exists(clusters_save_path_for_decay_matrix):
            layer_idx_to_clusters_for_decay_matrix = np.load(clusters_save_path_for_decay_matrix, allow_pickle=True).item()
        else:
            if local_rank == 0:
                # 获取聚类的 idx 结果
                # # 返回的是一个字典，每个 key 是层id, 对应的 value 的值是一个长度等于当前层长度的聚类结果。[[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
                # 生成 layer_idx_to_clusters_for_decay_matrix 为了计算 decay matrix
                layer_idx_to_clusters_for_decay_matrix = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                                  target_deps=target_deps_for_decay_matrix,
                                                                  pacesetter_dict=pacesetter_dict)
                # pacesetter_dict 是残差结构的连接层之间的关系。
                if pacesetter_dict is not None:
                    for follower_idx, pacesetter_idx in pacesetter_dict.items():
                        # 这里将残差的最后一层的剪枝方案直接等同于直连的剪枝方案
                        if pacesetter_idx in layer_idx_to_clusters_for_decay_matrix:
                            layer_idx_to_clusters_for_decay_matrix[follower_idx] = layer_idx_to_clusters_for_decay_matrix[pacesetter_idx]
                # 保存聚类的 idx 结果
                np.save(clusters_save_path_for_decay_matrix, layer_idx_to_clusters_for_decay_matrix)
            else:
                while not os.path.exists(clusters_save_path_for_decay_matrix):
                    time.sleep(10)
                    print('sleep, waiting for process 0 to calculate clusters')
                layer_idx_to_clusters_for_decay_matrix = np.load(clusters_save_path_for_decay_matrix, allow_pickle=True).item()
        # ----------------------------------- done -----------------------------------
        

        # 根据 聚类的通道的结果，生成 matrix，方便计算
        param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=deps,
                                                                      layer_idx_to_clusters=layer_idx_to_clusters_for_kernel_matrix,
                                                                      kernel_namedvalue_list=kernel_namedvalue_list)
        # 这块的功能似乎是要添加每层对于的 bias\gamma\beta 进入这个 param_name_to_merge_matrix
        add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix)
        # 加入超参centri_strength控制的向心力，作为新的 weight decay
        param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(
            deps=deps,
            layer_idx_to_clusters=layer_idx_to_clusters_for_decay_matrix,
            kernel_namedvalue_list=kernel_namedvalue_list,
            weight_decay=1e-4,
            weight_decay_bias=0,
            centri_strength=centri_strength)
        # ----------------------------------- done -----------------------------------

        # ------------------- 获取聚类 -------------------
        conv_idx = 0
        param_to_clusters = {} # 通过 layer_idx_to_clusters 获得 param_to_clusters
        for k, v in model.named_parameters():
            if v.dim() != 4:
                continue
            if conv_idx in layer_idx_to_clusters_for_decay_matrix:
                for clsts in layer_idx_to_clusters_for_decay_matrix[conv_idx]:
                    if len(clsts) > 1:
                        param_to_clusters[v] = layer_idx_to_clusters_for_decay_matrix[conv_idx]
                        break
            conv_idx += 1
        # ----------------------------------- done -----------------------------------

        start_epoch, end_epoch = 40, 200

        kwargs = {"param_name_to_merge_matrix":param_name_to_merge_matrix, "param_name_to_decay_matrix":param_name_to_decay_matrix}
        best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name+'-global-cluster', optimizer, scheduler, loss_func, 
                    best_acc, writer, start_epoch, end_epoch, is_update=True, param_to_clusters=param_to_clusters,layer_idx_to_clusters=layer_idx_to_clusters_for_decay_matrix, **kwargs)
        
        if local_rank == 0:
            print("Best Acc=%.4f" % (best_acc))

        # ---------------------------  全局聚类    ------------------------------
        schedule = 0.90 # 超参，控制 pca 拟合情况
        target_deps = generate_itr_for_model_follow_global_cluster(schedule, model)
        clusters_save_path = os.path.join(model_save_path, 'clusters.npy')
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
        param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(
            deps=deps,
            layer_idx_to_clusters=layer_idx_to_clusters,
            kernel_namedvalue_list=kernel_namedvalue_list,
            weight_decay=1e-4,
            weight_decay_bias=0,
            centri_strength=centri_strength)
        # ----------------------------------- done -----------------------------------

        # ------------------- 获取聚类 -------------------
        conv_idx = 0
        param_to_clusters = {} # 通过 layer_idx_to_clusters 获得 param_to_clusters
        for k, v in model.named_parameters():
            if v.dim() != 4:
                continue
            if conv_idx in layer_idx_to_clusters:
                for clsts in layer_idx_to_clusters[conv_idx]:
                    if len(clsts) > 1:
                        param_to_clusters[v] = layer_idx_to_clusters[conv_idx]
                        break
            conv_idx += 1
        # ----------------------------------- done -----------------------------------

        start_epoch, end_epoch = 200, args.total_epochs

        kwargs = {"param_name_to_merge_matrix":param_name_to_merge_matrix, "param_name_to_decay_matrix":param_name_to_decay_matrix}
        best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name+'-part-cluster', optimizer, scheduler, loss_func, 
                    best_acc, writer, start_epoch, end_epoch, is_update=True, param_to_clusters=param_to_clusters,layer_idx_to_clusters=layer_idx_to_clusters, **kwargs)
        
        if local_rank == 0:
            print("Best Acc=%.4f" % (best_acc))


def train_core(model, train_loader, test_loader, 
                model_save_path, model_name, optimizer, scheduler, loss_func, 
                best_acc, writer, start_epoch, end_epoch, 
                is_update=False,param_to_clusters=None,layer_idx_to_clusters=None, **kwargs):
    for epoch in range(start_epoch, end_epoch):
        model.train()
        for i, (img, target) in enumerate(train_loader):
            img, target = img.to(local_rank), target.to(local_rank)
            out = model(img)
            loss = loss_func(out, target)
            loss.backward()

            # update networl params based on the CSGD
            if is_update:
                update_net_params(model, **kwargs)

            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" %
                    (epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        if local_rank==0:
            print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))

        model_file = model_save_path + model_name +'-round%d.pth' % (args.round)
        if best_acc < acc and local_rank == 0:
            if  os.path.exists(model_file):
                os.remove(model_file)
            torch.save(model.module, model_file)
            best_acc = acc
        if (epoch % 20) == 19 and local_rank == 0:
            model_file = model_save_path + model_name +f'-{epoch}-round{args.round}.pth' 
            torch.save(model.module, model_file)
        scheduler.step()
        if writer != None:
            writer.add_scalar('eval/acc', acc, epoch)
            writer.add_scalar('train/loss', loss, epoch)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

        if is_update and local_rank == 0:
            deviation_sum = 0
            for param, clusters in param_to_clusters.items():
                pvalue = param.detach().cpu().numpy()
                for cl in clusters:
                    if len(cl) == 1:
                        continue
                    selected = pvalue[cl, :, :, :]
                    mean_kernel = np.mean(selected, axis=0, keepdims=True)
                    diff = selected - mean_kernel
                    deviation_sum += np.sum(diff ** 2)
            deviation_sum2 = calcu_sum_of_samplers_to_their_closest_cluster_center(model, layer_idx_to_clusters)
            if writer != None:
                writer.add_scalar('train/deviation_sum', deviation_sum, epoch)
                writer.add_scalar('train/deviation_sum2', deviation_sum2, epoch)

    return best_acc


def main():
    model_save_path = 'save/train_and_prune/' + TIMESTAMP
    tensorboard_log_path = TENSORBOARD_LOG_DIR + TIMESTAMP
    tos_model_save_path = '/tos/save_data/my_pruning_save_data/log_and_model/' + 'SGD_CAWR_test2_600epoch/' +'0.90-'+ TIMESTAMP
    tos_tensorboard_log_path = '/tos/save_data/my_pruning_save_data/log_and_model/' + 'SGD_CAWR_test2_600epoch/' +'0.90-'+ TIMESTAMP
    print(tos_model_save_path)

    if local_rank == 0:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(tensorboard_log_path, exist_ok=True)

    train_loader, test_loader = get_dataloader()

    if args.mode == 'train':
        args.round = 0
        model = ResNet50(num_classes=10)
        model_name = "ResNet50"
        train_model(model,model_name, train_loader, test_loader, model_save_path)
        if local_rank == 0:
            os.makedirs(tos_model_save_path, exist_ok=True)
            os.makedirs(tos_tensorboard_log_path, exist_ok=True)
            copy_files(model_save_path, tos_model_save_path)
            copy_files(tensorboard_log_path, tos_tensorboard_log_path)
    elif args.mode == 'train_with_csgd':
        args.round = 0
        model = ResNet50(num_classes=10)
        # previous_ckpt = 'save/train_and_prune/ResNet50-round0.pth'
        # print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        # model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        model_name = "ResNet50-CSGD"
        train_with_csgd(model, model_name, train_loader,test_loader,is_resume=True, model_save_path= model_save_path)
        if local_rank == 0:
            os.makedirs(tos_model_save_path, exist_ok=True)
            os.makedirs(tos_tensorboard_log_path, exist_ok=True)
            copy_files(model_save_path, tos_model_save_path)
            copy_files(tensorboard_log_path, tos_tensorboard_log_path)
    elif args.mode == 'prune':
        previous_ckpt = 'save/train_and_prune/ResNet50-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        prune_model(model)
        torch.save(model, 'save/train_and_prune/ResNet50-round%d.pth' % (args.round))
    elif args.mode == 'finetune':
        # previous_ckpt = 'save/train_and_prune/ResNet50-round%d.pth' % (args.round)
        # print("Finetune round %d, load model from %s" % (args.round, previous_ckpt))
        # model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        model_name = "ResNet50-CSGD-finetune"
        model = ResNet50(num_classes=10)

        train_model(model, model_name, train_loader, test_loader, model_save_path=model_save_path)
        if local_rank == 0:
            copy_file(model_save_path + model_name +'-round%d.pth' % (args.round), tos_model_save_path)
    elif args.mode == 'test':
        # # ckpt = 'save/train_and_prune/2022-02-16T00-42-55/ResNet50-finetune-round1.pth'
        # ckpt = "/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.0003_0.25_600epoch/ResNet50-CSGD-finetune-round1.pth"
        # print("Load model from %s" % (ckpt))
        # # need load model to cpu, avoid computing model GPU memory errors
        # model = torch.load(ckpt, map_location=torch.device("cpu"))

        # hdf5_file = f'/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.0003_0.25_600epoch/prune_mode.hdf5'
        # print(hdf5_file)
        # model = ResNet50(num_classes=10)
        # load_hdf5(model, hdf5_file)

        # from utils.misc import load_hdf5
        
        list1 = ['0.001', '0.003', '0.0003']
        list2 = ['0.25', '0.50', '0.75']
        for l in list1:
            for n in list2:
                # hdf5_file = f'save/move/{l}-{n}/prune_mode.hdf5'
                # if not os.path.exists(hdf5_file):
                #     continue
                # print(hdf5_file)
                # model = ResNet50(num_classes=10)
                # load_hdf5(model, hdf5_file)
                ckpt = f"/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_{l}_{n}_600epoch/ResNet50-CSGD-finetune-round1.pth"
                print("Load model from %s" % (ckpt))
                # # need load model to cpu, avoid computing model GPU memory errors
                model = torch.load(ckpt, map_location=torch.device("cpu"))
        
                fake_input = torch.randn(1, 3, 32, 32)
                
                model_infor = get_model_infor_and_print(model, fake_input, 0)

                model_infor['Model'] = 'resnet-50'
                start = time.time()
                model_infor['Top1-acc(%)'] = eval(model, test_loader)
                end = time.time()
                print("time is :", end-start)
                model_infor['Top5-acc(%)'] = ' '
                model_infor['If_base'] = 'False'
                # model_infor['Strategy'] = f'CSGD+{block_prune_probs}' if model_infor['If_base'] == 'False' else ' '
                model_infor['Strategy'] = f'SGD_CAWR_{l}_{n}_600epoch'
                print(model_infor)

                excel_path = "model_data_2.xlsx"
                read_excel_and_write(excel_path, model_infor)


        # fake_input = torch.randn(1, 3, 32, 32)
        # model_infor = get_model_infor_and_print(model, fake_input, 0)

        # model_infor['Model'] = 'resnet-50'
        # model_infor['Top1-acc(%)'] = eval(model, test_loader)
        # print(model_infor['Top1-acc(%)'])
        # model_infor['Top5-acc(%)'] = ' '
        # model_infor['If_base'] = 'False'
        # # model_infor['Strategy'] = f'CSGD+{block_prune_probs}' if model_infor['If_base'] == 'False' else ' '
        # model_infor['Strategy'] = f'SGD_CAWR_0.0003_0.25_600epoch'
        # print(model_infor)

        # excel_path = "model_data_2.xlsx"
        # read_excel_and_write(excel_path, model_infor)


if __name__ == '__main__':
    main()
