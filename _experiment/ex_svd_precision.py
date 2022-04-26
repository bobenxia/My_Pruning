import logging
import numbers
import torchvision
import torch
import numpy as np
from sklearn.decomposition import PCA
# import wandb


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

    logging.info('Accuracy of the network on the 10000 test images: %s %%' % (100 * correct / total))
    return correct / total

#numpy 实现 pca
def pca(X, k_percentage):
    # X: m*n
    m, n = X.shape
    # 均值归一化
    X = X - np.mean(X, axis=0)
    # 协方差矩阵
    cov = np.dot(X.T, X) / m
    # 取特征值和特征向量
    eig_value, eig_vector = np.linalg.eig(cov)
    # 将特征值从大到小排序
    index = np.argsort(-eig_value)
    sort_eig_value = np.sort(-eig_value)
    # print(index)
    sum_index_ls = np.cumsum(sort_eig_value) / np.sum(sort_eig_value)
    # print("sum_index_ls", sum_index_ls)
    sum_index_list = [j for j, m in enumerate(sum_index_ls) if m > k_percentage]
    k = sum_index_list[0] if sum_index_list else len(index)-1
    # print("k", k, "index", len(index))
    print(eig_value.shape, m,n, k)

    # 取前k个特征值对应的特征向量
    eig_value = eig_value[index][:k]
    eig_vector = eig_vector[:, index][:, :k]
    # 将数据转换到新空间
    X_pca = np.dot(X, eig_vector)
    return X_pca, eig_value, eig_vector,k



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


def pca_model(model, pca_k):
    model.cpu()
    weight_dict = {}
    channel_ls = []
    pca = PCA(n_components=pca_k, svd_solver="full")
    for k, v in model.state_dict().items():
        if len(v.shape) == 4:
            weight_origin = v.numpy()
            weight = np.reshape(weight_origin, (weight_origin.shape[0], -1))
            weight = np.transpose(weight)
            # weight = weight - np.mean(weight, axis=0)
            pca_res = pca.fit_transform(weight)
            number_of_channel = pca_res.shape[1]
            # print(number_of_channel)

            new_weight = pca_res.transpose()
            new_weight = np.reshape(new_weight, (-1, weight_origin.shape[1], weight_origin.shape[2], weight_origin.shape[3]))
            weight_dict[k] = new_weight
            channel_ls.append(number_of_channel)
    return weight_dict, channel_ls

# def pca_model(model, pca_k):
#     model.cpu()
#     weight_dict = {}
#     channel_ls = []
#     # pca = PCA(n_components=pca_k, svd_solver="full")
#     for k, v in model.state_dict().items():
#         if len(v.shape) == 4:
#             weight_origin = v.numpy()
#             weight = np.reshape(weight_origin, (weight_origin.shape[0], -1))
#             weight = np.transpose(weight)
#             # pca_res = pca.fit_transform(weight)
#             x_pca, eig_value, eig_vector,number_of_channel = pca(weight, pca_k)

#             # new_weight = pca_res.transpose()
#             # new_weight = np.reshape(new_weight, (-1, weight_origin.shape[1], weight_origin.shape[2], weight_origin.shape[3]))
#             # weight_dict[k] = new_weight
#             channel_ls.append(number_of_channel)
#             weight_dict = 1
#     return weight_dict, channel_ls

def test_svd_mode(svd_k_ls):
    f2_ls = []
    acc_ls = []
    channel_pca_ls_total = []
    channel_pca_ls_part = []
    channel_svd_ls_total = []
    channel_svd_ls_part = []
    channel_origin_ls = []
    model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.0_1.0_600epoch/ResNet50-round0.pth", map_location=torch.device("cpu"))
    model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
    for k, v in model.state_dict().items():
        if len(v.shape) == 4:
            weight_origin = v.numpy()
            channel_origin_ls.append(weight_origin.shape[0])
    for svd_k in svd_k_ls:
        model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.0_1.0_600epoch/ResNet50-round0.pth", map_location=torch.device("cpu"))
        model = torch.load("/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-21T14-40-21/ResNet50-CSGD-global-cluster-199-round0.pth", map_location=torch.device("cpu"))
        weight_dict_svd, channel_ls_svd, f2 = svd_model(model, svd_k)
        weight_dict_pca, channel_ls_pca = pca_model(model, svd_k)
        # for k, v in model.named_parameters():
        #     if len(v.shape) == 4:
        #         v.data = torch.from_numpy(weight_dict[k])
        # acc = eval_model(model, dataloader_test)
        f2_ls.append(f2)
        # acc_ls.append(acc)
        # wandb.log({'acc': acc}, step=int(svd_k*100))
        # wandb.log({'f2': f2}, step=int(svd_k*100))
        pca_result_part_ls = [(channel_origin_ls[i] - channel_ls_pca[i])/channel_origin_ls[i] for i in range(len(channel_ls_pca))]
        pca_result_part = sum(pca_result_part_ls) / len(channel_ls_pca)
        channel_pca_ls_part.append(pca_result_part)

        pca_result_total = (sum(channel_origin_ls) - sum(channel_ls_pca))/sum(channel_origin_ls)
        channel_pca_ls_total.append(pca_result_total)

        svd_result_part_ls = [(channel_origin_ls[i] - channel_ls_svd[i])/channel_origin_ls[i] for i in range(len(channel_ls_svd))]
        svd_result_part = sum(svd_result_part_ls) / len(channel_ls_svd)
        channel_svd_ls_part.append(svd_result_part)

        svd_result_total = (sum(channel_origin_ls) - sum(channel_ls_svd))/sum(channel_origin_ls)
        channel_svd_ls_total.append(svd_result_total)

        print("svd_result_part", svd_result_part)
        print("svd_result_total", svd_result_total)
        print("pca_result_part", pca_result_part)
        print("pca_result_total", pca_result_total)
    
    print("channel_pca_ls_total: ", channel_pca_ls_total)
    print("channel_pca_ls_part: ", channel_pca_ls_part)
    print("channel_svd_ls_total: ", channel_svd_ls_total)
    print("channel_svd_ls_part: ", channel_svd_ls_part)
    print(f2_ls)
    print(acc_ls)
    return f2_ls, acc_ls





if __name__ == "__main__":
    # wandb.init(project="ex_svd_model_precision")
    # wandb.run.name = "2022_04_09_2"
    ls = np.concatenate((np.arange(0.1, 0.70, 0.05), np.arange(0.70, 0.99, 0.02), np.arange(0.98, 0.999, 0.002)))
    ls = [0.99]
    # print(list(ls))
    f2_ls, acc_ls = test_svd_mode(ls)

    # a = np.array([[1,2,3],[4,5,6], [2,4,6], [2,4,6], [2,5,6]])
    # a = a.transpose()

    # x_pca, eig_value, eig_vector,k = pca(a, 1)
    # pca = PCA(n_components=0.999, svd_solver="full")
    # pca_res = pca.fit_transform(a)
    # print(pca_res.shape[1])

    # print(k)

