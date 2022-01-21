import numpy as np
import torch
from sklearn.cluster import KMeans


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
    result = []  # record result
    for j in range(num_cluster):
        result.append([])
    for i, c in enumerate(km.labels_):
        result[c].append(i)
    for r in result:
        assert len(r) > 0
    return result


def _is_follower(layer_idx, pacesetter_dict):
    followers_and_pacesetters = set(pacesetter_dict.keys())
    return (layer_idx in followers_and_pacesetters) and (pacesetter_dict[layer_idx] != layer_idx)


def get_layer_idx_to_clusters(kernel_namedvalue_list, target_deps, pacesetter_dict):
    # 返回的是一个字典，每个 key 对应的 value 的值是一个长度等于当前层长度的聚类结果。[[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
    result = {}
    for layer_idx, named_kv in enumerate(kernel_namedvalue_list):
        num_filters = named_kv.value.shape[0]

        if pacesetter_dict is not None and _is_follower(layer_idx, pacesetter_dict):
            continue

        if num_filters > target_deps[layer_idx]:
            # print(named_kv.name)
            result[layer_idx] = cluster_by_kmeans(kernel_value=named_kv.value, num_cluster=target_deps[layer_idx])
        elif num_filters < target_deps[layer_idx]:
            raise ValueError('wrong target dep')
    return result


def generate_itr_to_target_deps_by_schedule_vector(schedule, origin_deps, internal_kernel_idxes):
    """
        schedule 是通道保留比例
        origin_deps 是模型卷积通道列表
        internal_kernel_idxes 残差内部的通道
        
        后面两个参数都需要人为查看模型，记录得到的
    """
    # TODO: 现在没有考虑残差结构内外部剪枝不同的情况，全部按照一个比例剪枝
    final_deps = np.array(origin_deps)
    for i in range(1, len(origin_deps)):  # starts from 0 if you want to prune the first layer
        # if i in internal_kernel_idxes:
        final_deps[i] = np.ceil(final_deps[i] * schedule).astype(np.int32)
        # else:
    return final_deps


def generate_merge_matrix_for_kernel(deps, layer_idx_to_clusters, kernel_namedvalue_list):
    result = {}
    for layer_idx, clusters in layer_idx_to_clusters.items():
        # 每层的通道数目
        num_filters = deps[layer_idx]
        # 构建 num_filters * num_filters 0的矩阵
        merge_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        # 距离 clusters, 16 聚类 4 的结果 [[1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]]
        for clst in clusters:
            # 此时 clst 分别是 [1, 10, 11, 12, 14], [3, 6], [0, 4, 7, 8, 9, 13], [2, 5, 15]
            if len(clst) == 1:
                merge_trans_mat[clst[0], clst[0]] = 1
                continue
            sc = sorted(clst)  # Ideally, clst should have already been sorted in ascending order
            for ei in sc:
                for ej in sc:
                    merge_trans_mat[ei, ej] = 1 / len(clst)
        result[kernel_namedvalue_list[layer_idx].name] = torch.from_numpy(merge_trans_mat).cuda()
        # 这样每层都能得到一个 聚类后id 的 matrix
        # 这个 matrix 是为了加快计算用的
    return result
