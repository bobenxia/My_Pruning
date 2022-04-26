import logging
import numbers
from tabnanny import check
import torchvision
import torch
import numpy as np
from sklearn.decomposition import PCA
# import wandb
from pytorch_resnet_cifar10 import resnet
from utils.stagewise_resnet import create_SRC56,create_SRC110
from utils.constant import rc_pacesetter_dict
from utils.builder import ConvBuilder
from utils.constant import rc_origin_deps_flattened, rc_internal_layers


logging.basicConfig(level=logging.INFO)
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader_test = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
print(len(dataloader_test))


def eval_model(model, dataloader_test):
    model.eval().cuda()
    correct = 0
    total = 0
    for i, (data, target) in enumerate(dataloader_test):
        data = data.cuda()
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True).cpu()
        total += 1
        correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / total


def svd_model(model, svd_k):
    model.cpu()
    weight_dict = {}
    channel_ls = []
    for k, v in model.state_dict().items():
        if len(v.shape) == 4:
            weight_origin = v.numpy()
            # weight_origin = weight_origin - np.mean(weight_origin, axis=0)
            weight = np.reshape(weight_origin, (weight_origin.shape[0], -1))
            # weight = weight /min(weight.shape)
            weight = np.transpose(weight)

            v, s, vh = np.linalg.svd(weight, full_matrices=False)

            s_val = np.square(s) / (weight.shape[0] - 1)
            s_sum_list =  np.cumsum(s_val) / np.sum(s_val)
            s_sum_list_idx = [j for j, m in enumerate(s_sum_list) if m > svd_k]
            idx = s_sum_list_idx[0] if s_sum_list_idx else len(s)-1
            number_of_channel = idx
            # print(s.shape,idx)
            
            s_copy = s.copy()
            s_copy[idx:] = 0

            new_weight = np.dot(v, np.dot(np.diag(s_copy), vh))
            diff = weight - new_weight
            f2 = np.linalg.norm(diff)
            # logging.info("name:{}, origin channel: {} , new channel: {} , f2: {}".format(k, len(s), idx, np.linalg.norm(diff)))
            # wandb.log({f'{k}': idx}, step=int(svd_k*100))

            new_weight = np.transpose(new_weight)
            new_weight = np.reshape(new_weight, (weight_origin.shape[0], weight_origin.shape[1], weight_origin.shape[2], weight_origin.shape[3]))
            weight_dict[k] = new_weight   
            channel_ls.append(number_of_channel)
    return weight_dict, channel_ls, f2


def test_svd_mode_vgg(svd_k_ls):
    acc_ls = []
    # model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
    model = torch.load("save/train_and_prune/2022-04-18T20-32-52/vgg-599-round1.pth", map_location=torch.device("cpu"))
    print(eval_model(model, dataloader_test))
    for svd_k in svd_k_ls:
        # model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
        model = torch.load("save/train_and_prune/2022-04-18T20-32-52/vgg-599-round1.pth", map_location=torch.device("cpu"))
        weight_dict, _,_ = svd_model(model, svd_k)
        for k, v in model.named_parameters():
            if len(v.shape) == 4:
                v.data = torch.from_numpy(weight_dict[k])
        acc = eval_model(model, dataloader_test)
        logging.info('Accuracy of the network on the test images by%s: %s' % (svd_k,100 * acc))
        acc_ls.append(acc)
    print(acc_ls)
    return acc_ls

def test_svd_mode_resnet56(svd_k_ls):
    acc_ls = []
    deps = rc_origin_deps_flattened(9)
    convbuilder = ConvBuilder(base_config=None)
    model = create_SRC56(deps ,convbuilder)
    checkpoint = torch.load("save/resnet56-CSGD-part-cluster-599-round0.pth")
    model.load_state_dict(checkpoint["state_dict"])
    print(eval_model(model, dataloader_test))
    for svd_k in svd_k_ls:
        # model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        weight_dict, _,_ = svd_model(model, svd_k)
        for k, v in model.named_parameters():
            if len(v.shape) == 4:
                v.data = torch.from_numpy(weight_dict[k])
        acc = eval_model(model, dataloader_test)
        logging.info('Accuracy of the network on the test images by%s: %s' % (svd_k,100 * acc))
        acc_ls.append(acc)
    print(acc_ls)
    return acc_ls

def test_svd_mode_resnet110(svd_k_ls):
    acc_ls = []
    deps = rc_origin_deps_flattened(18)
    convbuilder = ConvBuilder(base_config=None)
    model = create_SRC110(deps ,convbuilder)
    checkpoint = torch.load("save/train_and_prune/2022-04-19T18-40-07/ResNet110-round0.pth")
    model.load_state_dict(checkpoint["state_dict"])
    print(eval_model(model, dataloader_test))
    for svd_k in svd_k_ls:
        # model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        weight_dict, _,_ = svd_model(model, svd_k)
        for k, v in model.named_parameters():
            if len(v.shape) == 4:
                v.data = torch.from_numpy(weight_dict[k])
        acc = eval_model(model, dataloader_test)
        logging.info('Accuracy of the network on the test images by%s: %s' % (svd_k,100 * acc))
        acc_ls.append(acc)
    print(acc_ls)
    return acc_ls



if __name__ == "__main__":
    # wandb.init(project="ex_svd_model_precision")
    # wandb.run.name = "2022_04_09_2"
    ls = np.concatenate((np.arange(0.1, 0.70, 0.05), np.arange(0.70, 0.99, 0.02)))#, np.arange(0.98, 0.999, 0.002)))
    # # print(list(ls))
    # ls = [0.99]
    # acc_ls = test_svd_mode_vgg(ls)
    acc_ls = test_svd_mode_resnet56(ls)
    # acc_ls = test_svd_mode_resnet110(ls)


