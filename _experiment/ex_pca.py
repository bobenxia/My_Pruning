#%%
from os import scandir
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


import torchvision
import numpy as np
import matplotlib.pyplot as plt

model = torchvision.models.resnet34(pretrained=True)


# # 找出 list 中大于i的第一个元素的索引
# idx = lambda i, l: [j for j, x in enumerate(l) if x > i][0]

def calcu_x(weight, ls:list):
    weight = np.reshape(weight, (weight.shape[0], -1))
    weight = np.transpose(weight)

    x = []
    y = []
    z = []
    for i in ls:
        # 找出递增列表sum_ls中元素大于i的最小索引
        model = PCA(n_components=i).fit(weight)
        w = model.transform(weight)
        print(w.shape)

        x.append(i)
        z.append(weight.shape[0] - w.shape[1])

    # plt.plot(x, y)
    # plt.show()

    plt.plot(x, z)
    plt.show()
        
ls = np.concatenate((np.arange(0.1, 0.90, 0.05), np.arange(0.90, 0.99, 0.01)))

for k, v in model.state_dict().items():
    if len(v.shape) == 4:
        weight = v.cpu().numpy()
        weight = np.reshape(weight, (weight.shape[0], -1))
        weight = np.transpose(weight)

        calcu_x(weight, ls)

        break


# %%
