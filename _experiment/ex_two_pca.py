#%%
from math import sqrt
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans
from torchvision import transforms

def set_seed(seed):
    torch.manual_seed(seed)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed(seed)  # gpu 为当前GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)

set_seed(1)



model = torch.load("save/train_and_prune/ResNet50-CSGD-round0.pth", map_location=torch.device("cpu"))
list1 = []
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
        list1.append(m)
# for i in range(len(list1)):
#     print(i, list1[i].weight.shape)


# torch 定义两层卷积的网络
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        # self.conv1 = list1[0]
        self.conv1 = torch.nn.Conv2d(3, 64, 5, 5, bias=False)
        # self.conv2 = list1[4]
        # self.conv3 = list1[12]
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 3, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 16, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1,x2, x3


model = Net().cuda().eval()
for k, v in model.named_parameters():
    print(v.data.shape, v.data[0,0,0,:])

# img = torch.load("./_experiment/img.pth")
# 读取图片
img = plt.imread("/root/xiazheng/panonet_onboard_use/20211218_school/My_Pruning/_experiment/dog_3.jpg")
# 标准化图片
transform_test = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.Resize((2048,2048)),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
img = transform_test(img)
img = img.unsqueeze(0)
# print(img.shape)

out1, out2, out3 = model(img.cuda())
print(img.shape, out1.shape, out2.shape, out3.shape)

# plt.figure(1)
# for i in range(16):
#     plt.subplot(4,4,  i + 1)
#     plt.axis('off')
#     plt.imshow(out1[0, i, :, :].cpu().detach(), cmap='gray')
# plt.show()

sum_ls = {}
for i in range(16):
    ls = out3[0, i, :, :].cpu().detach().numpy()
    sum_ls[i] = np.sum(ls)
sorted_ls = sorted(sum_ls.items(), key=lambda x: x[1], reverse=True)
print(sorted_ls)
img_dict = []
for i in range(16):
    img_dict.append(out3[0, sorted_ls[i][0], :, :].cpu().detach().numpy())
img_dict = np.array(img_dict)
# print(img_dict.shape)
    
plt.figure(2)
for i in range(16):
    plt.subplot(4,4,  i + 1)
    plt.axis('off')
    # plt.imshow(out3[0, i, :, :].cpu().detach(), cmap='gray')
    plt.imshow(img_dict[i], cmap='gray')
plt.show()

for k, v in model.state_dict().items():
    if k == "conv1.weight":
        pca = PCA(n_components=0.99999)
        weight0 = v.cpu().numpy()
        weight1 = np.reshape(weight0, (weight0.shape[0], -1))
        weight1 = np.transpose(weight1)
        pca_res = pca.fit_transform(weight1)
        print("pca_res shape:", pca_res.shape)
        numbers = pca_res.shape[1]
        # numbers = 60
        print("numbers:", numbers)

        # weight2 = np.transpose(pca_res)
        # weight2 = np.reshape(weight2, (-1, weight0.shape[1], weight0.shape[2], weight0.shape[3]))
        # conv1_weight = weight2
        km = KMeans(n_clusters=numbers, n_init=40, max_iter=10000, tol=1e-18)
        weight2 = np.reshape(weight0, (weight0.shape[0], -1))
        km.fit(weight2)
        print("kmeans.n_iter_:", km.n_iter_)
        conv1_weight = np.reshape(km.cluster_centers_, (-1, weight0.shape[1], weight0.shape[2], weight0.shape[3]))
    elif k == "conv2.weight":
        km = KMeans(n_clusters=numbers, n_init=40, max_iter=10000, tol=1e-18)
    #     pca = PCA(n_components=0.999)
        weight = v.cpu().numpy()
        new_weight = None
        for i in range(weight.shape[0]):
            weight0 = weight[i, :, :, :]
            # print(weight0.shape)
            weight1 = np.reshape(weight0, (weight0.shape[0], -1))
            km.fit(weight1)
            print(f"{i}, kmeans.n_iter_:", km.n_iter_)
            weight2 = np.reshape(km.cluster_centers_, (-1, weight0.shape[1], weight0.shape[2]))
            weight2 = weight2[np.newaxis, :, :, :]
            # print(weight2.shape)
            if new_weight is None:
                new_weight = weight2
            else:
                new_weight = np.concatenate((new_weight, weight2), axis=0)
        # print(new_weight.shape)






model = torch.load("save/train_and_prune/ResNet50-CSGD-round0.pth", map_location=torch.device("cpu"))
list1 = []
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
        list1.append(m)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        # self.conv1 = list1[0]
        self.conv1 = torch.nn.Conv2d(3, 64, 5, 5, bias=False)
        # self.conv2 = list1[4]
        # self.conv3 = list1[12]
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 3, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 16, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1,x2, x3



model = Net().cuda().eval()
for k, v in model.named_parameters():
    print(v.data.shape, v.data[0,0,0,:])

for k, v in model.named_parameters():
    if k == "conv1.weight":
        v.data = torch.from_numpy(conv1_weight).cuda().type(torch.cuda.FloatTensor)
        # v.data = v.data[0:numbers, :, :, :]
    elif k == "conv2.weight":
        v.data = torch.from_numpy(new_weight).cuda().type(torch.cuda.FloatTensor)
        # v.data = v.data[:, 0:numbers, :, :]
for k, v in model.named_parameters():
    print(v.data.shape, v.data[0,0,0,:])
out1, out2, out3 = model(img.cuda())
print(img.shape, out1.shape, out2.shape, out3.shape)

# numpy sum
sum_ls = {}
for i in range(16):
    ls = out3[0, i, :, :].cpu().detach().numpy()
    sum_ls[i] = np.sum(ls)
sorted_ls = sorted(sum_ls.items(), key=lambda x: x[1], reverse=True)
print(sorted_ls)
img_dict = []
for i in range(16):
    img_dict.append(out3[0, sorted_ls[i][0], :, :].cpu().detach().numpy())
img_dict = np.array(img_dict)
# print(img_dict.shape)
    
plt.figure(2)
for i in range(16):
    plt.subplot(4,4,  i + 1)
    plt.axis('off')
    # plt.imshow(out3[0, i, :, :].cpu().detach(), cmap='gray')
    plt.imshow(img_dict[i], cmap='gray')
plt.show()




model = torch.load("save/train_and_prune/ResNet50-CSGD-round0.pth", map_location=torch.device("cpu"))
list1 = []
for m in model.modules():
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.BatchNorm2d):
        list1.append(m)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(10)
        # self.conv1 = list1[0]
        self.conv1 = torch.nn.Conv2d(3, 64, 5, 5, bias=False)
        # self.conv2 = list1[4]
        # self.conv3 = list1[12]
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 3, bias=False)
        self.conv3 = torch.nn.Conv2d(64, 16, 1, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        return x1,x2, x3



model = Net().cuda().eval()
for k, v in model.named_parameters():
    print(v.data.shape, v.data[0,0,0,:])
for k, v in model.named_parameters():
    if k == "conv1.weight":
        # v.data = torch.from_numpy(conv1_weight).cuda().type(torch.cuda.FloatTensor)
        v.data = v.data[0:numbers, :, :, :]
    elif k == "conv2.weight":
        # v.data = torch.from_numpy(new_weight).cuda().type(torch.cuda.FloatTensor)
        v.data = v.data[:, 0:numbers, :, :]
for k, v in model.named_parameters():
    print(v.data.shape, v.data[0,0,0,:])
out1, out2, out3 = model(img.cuda())
print(img.shape, out1.shape, out2.shape, out3.shape)


# numpy sum
sum_ls = {}
for i in range(16):
    ls = out3[0, i, :, :].cpu().detach().numpy()
    sum_ls[i] = np.sum(ls)
sorted_ls = sorted(sum_ls.items(), key=lambda x: x[1], reverse=True)
print(sorted_ls)
img_dict = []
for i in range(16):
    img_dict.append(out3[0, sorted_ls[i][0], :, :].cpu().detach().numpy())
img_dict = np.array(img_dict)
# print(img_dict.shape)
    
plt.figure(3)
for i in range(16):
    plt.subplot(4,4,  i + 1)
    plt.axis('off')
    # plt.imshow(out3[0, i, :, :].cpu().detach(), cmap='gray')
    plt.imshow(img_dict[i], cmap='gray')
plt.show()



2# %%

# %%
