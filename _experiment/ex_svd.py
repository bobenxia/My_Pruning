#%%
import torchvision
import numpy as np
import matplotlib.pyplot as plt

model = torchvision.models.resnet34(pretrained=True)


# # 找出 list 中大于i的第一个元素的索引
# idx = lambda i, l: [j for j, x in enumerate(l) if x > i][0]

def calcu_x(weight, ls:list):
    weight = np.reshape(weight, (weight.shape[0], -1))
    weight = np.transpose(weight)
    v, s, vh = np.linalg.svd(weight,full_matrices=False)
    print(s)
    sum_ls = np.array([ np.sum(s[:i+1]) / len(s) for i in range(len(s)) ])
    print(sum_ls)

    x = []
    y = []
    z = []
    for i in ls:
        # 找出递增列表sum_ls中元素大于i的最小索引
        idx_ls = [j for j, m in enumerate(sum_ls) if m < i]
        idx = idx_ls[-1]
        print(idx)
        s_copy = s.copy()
        s_copy[idx:] = 0

        r = np.dot(v, np.dot(np.diag(s_copy), vh))
        diff = weight - r
        x.append(i)
        y.append(np.linalg.norm(diff))
        z.append(len(s) - idx)

    plt.plot(x, y)
    plt.show()

    plt.plot(x, z)
    plt.show()
        
ls = np.concatenate((np.arange(0.1, 0.90, 0.05), np.arange(0.90, 0.999, 0.001)))

for k, v in model.state_dict().items():
    if len(v.shape) == 4:
        weight = v.cpu().numpy()
        weight = np.reshape(weight, (weight.shape[0], -1))
        weight = np.transpose(weight)

        calcu_x(weight, ls)

        break


# %%
