from curses import resetty
import torch
import os
import torchvision
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


# 字典的值是一个列表，将列表中的元素逐个追加到excel的键对应的行中
def write_dict_to_excel_append_list(dict, excel_path):
    if not os.path.exists(excel_path):
        df = pd.DataFrame(dict)
        df.to_excel(excel_path, index=False)
    else:
        df = pd.read_excel(excel_path)
        for k, v in dict.items():
            df[k].append(v)
        df.to_excel(excel_path, index=False)

def pca_for_weights(weight, name, schedule_list):
    print(name)
    result = []
    weight = weight.cpu().numpy()
    print(weight.shape)
    weight = np.reshape(weight, (weight.shape[0], -1))
    print(weight.shape)
    weight = np.transpose(weight)
    print(weight.shape)
    for schedule in schedule_list:
        pca = PCA(n_components=schedule)
        pca_res = pca.fit_transform(weight)
        result.append(pca_res.shape[1])
    result.append(weight.shape[1])
    print(result)
    return {name: result}


def generate_itr_for_model_follow_global_cluster(schedule, model):
    pca = PCA(n_components=schedule)
    result = []
    for k, v in model.state_dict().items():
        weight = v.cpu().numpy()
        if len(weight.shape) == 4:
            if len(result) == 0: # 跳过第一个卷积层，一般不剪枝
                result.append(weight.shape[0])
            else:
                print(weight.shape)
                weight = np.reshape(weight, (weight.shape[0], -1))
                print(weight.shape)
                weight = np.transpose(weight)
                print(weight.shape)
                pca_res = pca.fit_transform(weight)
                num_channel = int(pca_res.shape[1]/16+0.5) * 16 if int(pca_res.shape[1]/16+0.5) * 16 != 0 else 16
                result.append(num_channel)
                print(k,":    ",weight.shape[1]," -> ", pca_res.shape[1]," -> ", num_channel)
    return np.array(result)

# ckpt = '/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.003_0.75_600epoch/ResNet50-CSGD-round0.pth'
def print_pca_result():
    model = torchvision.models.resnet34(pretrained=True)
    # res = generate_itr_for_model_follow_global_cluster(0.95, model)
    a = np.arange(0.01, 0.99, 0.01)
    b = np.arange(0.99, 0.999, 0.001)
    schedule_list = np.append(a, b)
    result = {}
    # 字典追加

    for k, v in model.state_dict().items():
        if len(v.shape) == 4:
            r = pca_for_weights(v, k, schedule_list)
            result.update(r)
    print(result)
    write_dict_to_excel_append_list(result, './pca_result_resnet34.xlsx')


# print_pca_result()
model = torchvision.models.resnet50(pretrained=True)
count = 0
for k, v in model.state_dict().items():
    if len(v.shape) == 4:
        count += 1
        print(k, v.shape)
print(count)


def print_first_conv_weight():
    pca = PCA(n_components=0.99)
    model = torchvision.models.resnet50(pretrained=True)
    a = 0
    for k, v in model.state_dict().items():
        if len(v.shape) == 4 and k == "layer1.0.conv2.weight":
            # if a < 2:
            #     a += 1
            #     continue
            print(k, v.shape)
            weight = v.cpu().numpy()
            weight1 = np.reshape(weight, (weight.shape[0], -1))
            weight2 = np.transpose(weight1)
            pca_res = pca.fit_transform(weight2)
            print(pca_res.shape)

            km = KMeans(n_clusters=pca_res.shape[1])
            km.fit(weight1)
            print(list(km.labels_), km.inertia_)

            ls = list(km.labels_)
            result = []
            for i in range(len(ls)):
                # 找出 list 中具体值的全部下标
                index = [j for j, x in enumerate(ls) if x == i]
                print(i, index)
                # list 合并
                result.extend(index)
                # result.append(index)
            print(result)

            weight3 = weight.reshape(weight.shape[0],weight.shape[1], -1)
            print(weight3.shape)
            for i in range(64):
                plt.subplot(4, 16, i+1)
                plt.axis('off')
                plt.imshow(weight3[result[i]], cmap='gray')
                plt.savefig("./_experiment/ex_draw_kernel.png")
            plt.show()
            break
print_first_conv_weight()

