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
# from pytorch_resnet_cifar10.resnet import resnet110
from tools.print_model_info import get_model_infor_and_print
from tools.write_excel import read_excel_and_write
from utils.model_utils import *
from utils.misc import copy_files
from utils.stagewise_resnet import create_SRC110
from utils.builder import ConvBuilder
from utils.constant import rc_origin_deps_flattened

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

parser = argparse.ArgumentParser()
parser.add_argument('--mode',
                    type=str,
                    required=True,
                    choices=['train', 'prune', 'test', 'finetune', 'train_with_csgd'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=600)
parser.add_argument('--step_size', type=int, default=30)
parser.add_argument('--round', type=int, default=1)
parser.add_argument('--pruned_per', type=float, default=0.125)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument('--scheduler_', type=str, default='CAWR')
parser.add_argument('--out_dir', type=str, default='save/')

args = parser.parse_args()
local_rank = args.local_rank
block_prune_probs = [args.pruned_per] * 16
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
TENSORBOARD_LOG_DIR = os.getenv("TENSORBOARD_LOG_PATH", "/tensorboard_logs/")



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


def train_model(model,model_name, train_loader, test_loader,model_save_path):
    # DDP???DDP backend?????????
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl???GPU????????????????????????????????????

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    if args.scheduler_ == "ELR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler_ == "CAWR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0, T_0=args.total_epochs//3+1, T_mult=2)
    else:
        raise ValueError("scheduler type error!")
    loss_func = nn.CrossEntropyLoss().to(local_rank)


    model.to(local_rank)

    # DDP: ??????DDP model
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
            # ?????????????????????????????????reshape ??? g_mat
            # ???????????? g_mat ?????????????????????????????????????????????????????????
            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(
                param_mat)
            # ?????????????????????????????????????????????
            param.grad.copy_(csgd_gradient.reshape(p_size))


def train_with_csgd(model, model_name, train_loader, test_loader, is_resume, model_save_path):
    # -------------------- ????????????????????????????????? -------------------
    schedule = 0.75
    deps = RESNET50_ORIGIN_DEPS_FLATTENED  # resnet50 ??? ????????????
    target_deps = generate_itr_to_target_deps_by_schedule_vector(schedule, RESNET50_ORIGIN_DEPS_FLATTENED, RESNET50_INTERNAL_KERNEL_IDXES)
    pacesetter_dict = {
        4: 4,3: 4,7: 4,10: 4,14: 14,
        13: 14,17: 14,20: 14,23: 14,27: 27,26: 27,30: 27,33: 27,36: 27,
        39: 27,42: 27,46: 46,45: 46,49: 46,52: 46
    }
    # --------------------------- done ----------------------------

    # ------------ parepare optimizer, scheduler, criterion -------
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0.004, T_0=args.total_epochs//3+1, T_mult=2)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, eta_min=0.004, T_0=args.total_epochs+1, T_mult=2)
    loss_func = nn.CrossEntropyLoss().to(local_rank)
    centri_strength=0.02
    # --------------------------- done ------------------------------

    # ------------------- DDP???DDP backend????????? --------------------
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # nccl???GPU????????????????????????????????????
    model.to(local_rank)
    # DDP: ??????DDP model
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    # -------------------------- done ------------------------------

    best_acc = -1

    # summarywriter
    if local_rank == 0:
        writer = SummaryWriter(TENSORBOARD_LOG_DIR + TIMESTAMP)
    else:
        writer = None

    # ------------------- ????????????????????????????????????total_epoch//3 ----------------
    if not is_resume:
        start_epoch, end_epoch = 0, args.total_epochs//3
        best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name, 
        optimizer, scheduler, loss_func, best_acc, writer, start_epoch, end_epoch)
    # ----------------------------------- done -----------------------------------

    with ModelUtils(local_rank=local_rank) as engine:
        engine.setup_log(name='train', log_dir=model_save_path, file_name='ResNet110-CSGD-log.txt')
        engine.register_state(scheduler=scheduler, model=model, optimizer=optimizer)

        # ------------------- prepare the clusters and matrices for  C-SGD -------------------
        kernel_namedvalue_list = engine.get_all_conv_kernel_namedvalue_as_list()

        clusters_save_path = os.path.join(model_save_path, 'clusters.npy')
        # clusters_save_path = 'save/train_and_prune/2022-02-08T17-37-47/clusters.npy'
        if os.path.exists(clusters_save_path):
            layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()
        else:
            if local_rank == 0:
                # ??????????????? idx ??????
                # # ????????????????????????????????? key ??????id, ????????? value ????????????????????????????????????????????????????????????[[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
                layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                                  target_deps=target_deps,
                                                                  pacesetter_dict=pacesetter_dict)
                # pacesetter_dict ?????????????????????????????????????????????
                if pacesetter_dict is not None:
                    for follower_idx, pacesetter_idx in pacesetter_dict.items():
                        # ?????????????????????????????????????????????????????????????????????????????????
                        if pacesetter_idx in layer_idx_to_clusters:
                            layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]
                # ??????????????? idx ??????
                np.save(clusters_save_path, layer_idx_to_clusters)
            else:
                while not os.path.exists(clusters_save_path):
                    time.sleep(10)
                    print('sleep, waiting for process 0 to calculate clusters')
                layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()

        # ?????? ????????????????????????????????? matrix???????????????
        param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=deps,
                                                                      layer_idx_to_clusters=layer_idx_to_clusters,
                                                                      kernel_namedvalue_list=kernel_namedvalue_list)
        # ???????????????????????????????????????????????? bias\gamma\beta ???????????? param_name_to_merge_matrix
        add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix)
        # core code ?????????????????????????????????????????? weight decay
        param_name_to_decay_matrix = generate_decay_matrix_for_kernel_and_vecs(
            deps=deps,
            layer_idx_to_clusters=layer_idx_to_clusters,
            kernel_namedvalue_list=kernel_namedvalue_list,
            weight_decay=1e-4,
            weight_decay_bias=0,
            centri_strength=centri_strength)
        # ----------------------------------- done -----------------------------------

        # ------------------- ???????????? -------------------
        conv_idx = 0
        param_to_clusters = {} # ?????? layer_idx_to_clusters ?????? param_to_clusters
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

        if not is_resume:
            start_epoch, end_epoch = args.total_epochs//3, args.total_epochs
        else:
            start_epoch, end_epoch = 0, args.total_epochs

        kwargs = {"param_name_to_merge_matrix":param_name_to_merge_matrix, "param_name_to_decay_matrix":param_name_to_decay_matrix}
        best_acc = train_core(model, train_loader, test_loader, model_save_path, model_name, optimizer, scheduler, loss_func, 
                    best_acc, writer, start_epoch, end_epoch, is_update=True, param_to_clusters=param_to_clusters,layer_idx_to_clusters=layer_idx_to_clusters, **kwargs)

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
            if i % 10 == 0 and args.verbose and local_rank == 0:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f" %
                    (epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        if local_rank == 0:
            print("Epoch %d/%d, Acc=%.4f" % (epoch, args.total_epochs, acc))

        model_file = model_save_path + model_name +'-round%d.pth' % (args.round)
        if best_acc < acc and local_rank == 0:
            best_acc = acc
            if  os.path.exists(model_file):
                os.remove(model_file)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_acc,
            }, model_file)
        if (epoch % 20) == 19 and local_rank == 0:
            model_file = model_save_path + model_name +f'-{epoch}-round{args.round}.pth' 
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_prec1': best_acc,
            }, model_file)
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
    tos_model_save_path = '/Tos/save_data/my_pruning_save_data/log_and_model/' + TIMESTAMP
    tos_tensorboard_log_path = '/Tos/save_data/my_pruning_save_data/log_and_model/' + TIMESTAMP

    if local_rank == 0:
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(tensorboard_log_path, exist_ok=True)

    train_loader, test_loader = get_dataloader()

    if args.mode == 'train':
        args.round = 0
        # model = resnet110()
        deps = rc_origin_deps_flattened(18)
        convbuilder = ConvBuilder(base_config=None)
        model = create_SRC110(deps ,convbuilder)
        model_name = "ResNet110"
        train_model(model,model_name, train_loader, test_loader, model_save_path)
        # if local_rank == 0:
        #     copy_files(model_save_path, tos_model_save_path)
        #     copy_files(tensorboard_log_path, tos_tensorboard_log_path)
    elif args.mode == 'train_with_csgd':
        args.round = 0
        # model = resnet110()
        deps = rc_origin_deps_flattened(18)
        convbuilder = ConvBuilder(base_config=None)
        model = create_SRC110(deps ,convbuilder)
        previous_ckpt = 'save/train_and_prune/ResNet110-round0.pth'
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        model_name = "ResNet110-CSGD"
        train_with_csgd(model, model_name, train_loader,test_loader,is_resume=True, model_save_path= model_save_path)
    elif args.mode == 'prune':
        previous_ckpt = 'save/train_and_prune/ResNet110-round%d.pth' % (args.round - 1)
        print("Pruning round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        prune_model(model)
        torch.save(model, 'save/train_and_prune/ResNet110-round%d.pth' % (args.round))
    elif args.mode == 'finetune':
        previous_ckpt = 'save/train_and_prune/ResNet110-round%d.pth' % (args.round)
        print("Finetune round %d, load model from %s" % (args.round, previous_ckpt))
        model = torch.load(previous_ckpt, map_location=torch.device("cpu"))
        train_model(model, train_loader, test_loader, summary_writer="./runs/prune_resnet50_cifar10_after_prune.log")
    elif args.mode == 'test':
        ckpt = 'save/train_and_prune/ResNet110-CSGD-round%d.pth' % (args.round)
        print("Load model from %s" % (ckpt))

        fake_input = torch.randn(1, 3, 32, 32)
        # need load model to cpu, avoid computing model GPU memory errors
        model = torch.load(ckpt, map_location=lambda storage, loc: storage)

        from utils.misc import load_hdf5
        # model = resnet110()
        deps = rc_origin_deps_flattened(18)
        convbuilder = ConvBuilder(base_config=None)
        model = create_SRC110(deps ,convbuilder)
        hdf5_file = "save/prune_mode.hdf5"
        load_hdf5(model, hdf5_file)
        
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
