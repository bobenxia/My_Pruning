from curses import resetty
import torch
import os
import numpy as np
from sklearn.decomposition import PCA

def generate_itr_for_model_follow_global_cluster(schedule, model):
    pca = PCA(n_components=schedule)
    result = []
    for k, v in model.state_dict().items():
        weight = v.cpu().numpy()
        if len(weight.shape) == 4:
            if len(result) == 0: # 跳过第一个卷积层，一般不剪枝
                result.append(weight.shape[0])
            else:
                weight = np.reshape(weight, (weight.shape[0], -1))
                pca_res = pca.fit_transform(weight)
                num_channel = int(pca_res.shape[1]/16+0.5) * 16
                result.append(num_channel)
                print(k,":    ",weight.shape[0]," -> ", pca_res.shape[1]," -> ", num_channel)
    return np.array(result)

# ckpt = '/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.003_0.75_600epoch/ResNet50-CSGD-round0.pth'
def print_pca_result(i):
    ckpt = f"save/train_and_prune/2022-02-19T13-32-09/ResNet50-CSGD-{i}-round0.pth"
    if os.path.exists(ckpt):
        print(ckpt)
    model = torch.load(ckpt, map_location=torch.device("cpu"))
    res = generate_itr_for_model_follow_global_cluster(0.95, model)
    print(res)

for i in range(199, 200, 20):
    print_pca_result(i)


