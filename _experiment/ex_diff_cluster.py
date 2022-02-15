import numpy as np
import torch
from sklearn.cluster import KMeans, AffinityPropagation


def cluster_by_kmeans(kernel_value, num_cluster):
    # 举例16 聚 4 的结果：result = [[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
    assert kernel_value.ndim == 4  # n,c,h,w
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))  # n, c*h*w
    if num_cluster == x.shape[0]:  # if num_cluster == n, result = [0, 1, ..., n]
        result = [[i] for i in range(num_cluster)]
        return result
    # else:
    # print('cluster {} filters into {} clusters'.format(x.shape[0], num_cluster))
    km = KMeans(n_clusters=num_cluster)  # use sklearn.cluster.KMeans to cluster kernel_value
    km.fit(x)
    print(km.labels_)

    result = []  # record result
    for j in range(num_cluster):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result

def cluster_by_ap(kernel_value, beta):
    assert kernel_value.ndim == 4  # n,c,h,w
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))  # n, c*h*w

    ap = AffinityPropagation(random_state=5,damping=0.6)
    ap.fit(x)
    print(ap.labels_)

    result = []  # record result
    for j in range(ap.labels_.max()+1):
        result.append([])
    for i, c in enumerate(ap.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result, ap.labels_.max()+1


if __name__ == "__main__":
    torch.manual_seed(3)
    net = torch.nn.Conv2d(256, 512, 3)
    for m in net.modules():
        torch.nn.init.xavier_normal_(m.weight,gain=1)

    for k, v in net.state_dict().items():
        print(k, v.shape)
        weight = v.detach().cpu().numpy()
        # print(weight)
        break

    result_kmeans = cluster_by_kmeans(weight, 6)
    # print(result_kmeans)

    result_ap, n_ap = cluster_by_ap(weight, 1)
    print(result_ap, n_ap)