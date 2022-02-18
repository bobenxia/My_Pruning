import torch
import os
import numpy as np
from sklearn.decomposition import PCA


ckpt = '/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_0.003_0.75_600epoch/ResNet50-CSGD-round0.pth'
if os.path.exists(ckpt):
    print(ckpt)
model = torch.load(ckpt, map_location=torch.device("cpu"))
for k, v in model.state_dict().items():
    if k == "layer4.1.conv2.weight":
        weight = v.cpu().numpy()

print(weight.shape)


pca = PCA(n_components=0.90)
weight = np.reshape(weight, (weight.shape[0], -1))
result = pca.fit_transform(weight)
print(result.shape[1])