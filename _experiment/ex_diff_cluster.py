import numpy as np
import sklearn
import torch
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, MeanShift
from sklearn.decomposition import PCA, FastICA


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
    # print(km.labels_)

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
    # print(ap.affinity_matrix_)
    # print(ap.labels_)

    result = []  # record result
    for j in range(ap.labels_.max()+1):
        result.append([])
    for i, c in enumerate(ap.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result, ap.labels_.max()+1


def cluster_by_dpsacan(kernel_value, beta):
    assert kernel_value.ndim == 4  # n,c,h,w
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))  # n, c*h*w

    dpscan = DBSCAN(eps = 1, min_samples = 1)
    dpscan.fit(x)
    # print(dpscan.labels_)

    result = []  # record result
    for j in range(dpscan.labels_.max()+1):
        result.append([])
    for i, c in enumerate(dpscan.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result, dpscan.labels_.max()+1


def cluster_by_meanshift(kernel_value, beta):
    assert kernel_value.ndim == 4  # n,c,h,w
    x = np.reshape(kernel_value, (kernel_value.shape[0], -1))  # n, c*h*w

    dpscan = MeanShift(bandwidth=0.29)
    dpscan.fit(x)
    # print(dpscan.labels_)

    result = []  # record result
    for j in range(dpscan.labels_.max()+1):
        result.append([])
    for i, c in enumerate(dpscan.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result, dpscan.labels_.max()+1


if __name__ == "__main__":
    torch.manual_seed(3)
    # import torchvision
    # model = torchvision.models.resnet50(pretrained=True)

    import os
    ckpt = f"save/train_and_prune/2022-02-19T13-32-09/ResNet50-CSGD-199-round0.pth"
    if os.path.exists(ckpt):
        print(ckpt)
    model = torch.load(ckpt, map_location=torch.device("cpu"))

    for k, v in model.state_dict().items():
        if k == "layer2.0.conv1.weight":
            weight = v.cpu().numpy()
    print(weight.shape)

    # net = torch.nn.Conv2d(512, 128, 3)
    # for m in net.modules():
    #     torch.nn.init.xavier_normal_(m.weight,gain=1)

    # for k, v in net.state_dict().items():
    #     # print(k, v.shape)
    #     weight = v.detach().cpu().numpy()
    #     # print(weight)
    #     break

    # result_kmeans = cluster_by_kmeans(weight, 6)
    # print(result_kmeans)

    # result_ap, n_ap = cluster_by_ap(weight, 1)
    # print( n_ap)

    # result_ap, n_dpscan = cluster_by_dpsacan(weight, 1)
    # print( n_dpscan)

    # result_ap, n_dpscan = cluster_by_meanshift(weight, 1)
    # print( n_dpscan)

    
    weight = np.reshape(weight, (weight.shape[0], -1))
    print(weight.shape)
    weight = np.transpose(weight)
    print(weight.shape)


    pca = PCA(n_components=0.9)
    result = pca.fit_transform(weight)
    print(result.shape)
    L1 = [n[0] for n in result]  
    L2 = [n[1] for n in result]
    print(len(L1), len(L2))


    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']

    # 根据主成分分析结果绘制散点图
    for i in range(len(colors)):
        x = result[:, 0][digits.target == i]
        y = result[:, 1][digits.target == i]
        plt.scatter(x, y, c=colors[i])

    # 设置图例，0-9用不同颜色表示
    plt.legend(digits.target_names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # 设置坐标标签
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    # 设置标题
    plt.title("PCA Scatter Plot")

    # 显示图形
    plt.show()
    # ica = FastICA()
    # weight = np.reshape(weight, (weight.shape[0], -1))
    # result = ica.fit_transform(weight)
    # print(result.shape[1])