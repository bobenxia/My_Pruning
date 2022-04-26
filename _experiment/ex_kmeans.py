import numpy as np
import torch
import torchvision
from sklearn.cluster import KMeans

# from utils.cluster_params import cluster_by_kmeans

net = torch.nn.Conv2d(3, 16, 3)
for k, v in net.state_dict().items():
    print(k, v.shape)
    weight = v.detach().cpu().numpy()
    break

x = np.reshape(weight, (weight.shape[0], -1))

km = KMeans(n_clusters=8)
km.fit(x)
print(km.labels_, km.cluster_centers_.shape)
print(km.cluster_centers_[0,:])
print(x[0,:])


# def calcu_sum_of_samplers_to_their_closest_cluster_center(model, layer_idx_to_clusters):
#     conv_idx = 0
#     sum = 0
#     for k, v in model.state_dict().items():
#         if v.dim() != 4:
#             continue
#         pvalue = np.reshape(v, (v.shape[0], -1))
#         if conv_idx in layer_idx_to_clusters:
#             for clsts in layer_idx_to_clusters[conv_idx]:
#                 num_clsts = len(clsts)
#                 if num_clsts > 1:
#                     km = KMeans(n_clusters=1)
#                     km.fit(pvalue[clsts,:])
#                     sum += km.inertia_
#         # print(k, sum)
#         conv_idx += 1

#     # print(sum)
#     return sum


# model = torchvision.models.resnet50(pretrained=True).cuda()
# clusters_save_path = 'save/train_and_prune/2022-02-08T17-37-47/clusters.npy'
# layer_idx_to_clusters  = np.load(clusters_save_path, allow_pickle=True).item()

# calcu_sum_of_samplers_to_their_closest_cluster_center(model, layer_idx_to_clusters)


# layer_idx_to_clusters = {0:cluster_by_kmeans(weight, 8)}
# print(layer_idx_to_clusters)

# calcu_sum_of_samplers_to_their_closest_cluster_center(net, layer_idx_to_clusters)
