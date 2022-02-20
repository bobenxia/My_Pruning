import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

        if num_filters >= target_deps[layer_idx]:
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
        final_deps[i] = max(np.ceil(final_deps[i] * schedule).astype(np.int32), 1)
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


#   Recently it is popular to cancel weight decay on vecs
def generate_decay_matrix_for_kernel_and_vecs(deps, layer_idx_to_clusters, kernel_namedvalue_list, weight_decay, weight_decay_bias,
                                              centri_strength):
    # weight_decay_bias 现在不用了，一般设置为0
    # centri_strength 是人为设置的超参，控制组内的点聚集的速度
    result = {}
    #   for the kernel
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                decay_trans_mat[ee, ee] = weight_decay + centri_strength
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength / len(clst)
        kernel_mat = torch.from_numpy(decay_trans_mat).cuda()
        result[kernel_namedvalue_list[layer_idx].name] = kernel_mat

    #   for the vec params (bias, beta and gamma), we use 0.1 * centripetal strength
    for layer_idx, clusters in layer_idx_to_clusters.items():
        num_filters = deps[layer_idx]
        decay_trans_mat = np.zeros((num_filters, num_filters), dtype=np.float32)
        for clst in clusters:
            sc = sorted(clst)
            for ee in sc:
                # Note: using smaller centripetal strength on the scaling factor of BN improve the performance in some of the cases
                decay_trans_mat[ee, ee] = weight_decay_bias + centri_strength * 0.1
                for p in sc:
                    decay_trans_mat[ee, p] += -centri_strength * 0.1 / len(clst)
        vec_mat = torch.from_numpy(decay_trans_mat).cuda()
        bias_name, gamma_name, beta_name = get_bias_gamma_and_beta_name(kernel_namedvalue_list[layer_idx].name)
        result[bias_name] = vec_mat
        result[gamma_name] = vec_mat
        result[beta_name] = vec_mat

    return result


def get_bias_gamma_and_beta_name(kernel_name):
    bias_name = kernel_name.replace('weight', 'bias')
    if 'conv' in kernel_name:
        gamma_name = kernel_name.replace('conv', 'bn')
        beta_name = kernel_name.replace('conv', 'bn').replace('weight', 'bias')
    elif 'downsample' in kernel_name:
        gamma_name = kernel_name.replace('downsample.0', 'downsample.1')
        beta_name = kernel_name.replace('downsample.0', 'downsample.1').replace('weight', 'bias')
    elif 'shortcut' in kernel_name:
        gamma_name = kernel_name.replace('shortcut.0', 'shortcut.1')
        beta_name = kernel_name.replace('shortcut.0', 'shortcut.1').replace('weight', 'bias')
    return bias_name, gamma_name, beta_name


def add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix):
    # 将 kernel 得到的 matrix 推广到 同一个块的 conv.bias, bn.weight, bn.bias
    kernel_names = set(param_name_to_merge_matrix.keys())
    for name in kernel_names:
        # bias_name = name.replace('weight', 'bias')
        # if 'conv' in name:
        #     gamma_name = name.replace('conv', 'bn')
        #     beta_name = name.replace('conv', 'bn').replace('weight', 'bias')
        # elif 'downsample' in name:
        #     gamma_name = name.replace('downsample.0', 'downsample.1')
        #     beta_name = name.replace('downsample.0', 'downsample.1').replace('weight', 'bias')
        bias_name, gamma_name, beta_name = get_bias_gamma_and_beta_name(name)
        param_name_to_merge_matrix[bias_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[gamma_name] = param_name_to_merge_matrix[name]
        param_name_to_merge_matrix[beta_name] = param_name_to_merge_matrix[name]


def calcu_sum_of_samplers_to_their_closest_cluster_center(model, layer_idx_to_clusters):
    conv_idx = 0
    sum = 0
    for k, v in model.state_dict().items():
        if v.dim() != 4:
            continue
        pvalue = np.reshape(v.detach().cpu().numpy(), (v.shape[0], -1))
        if conv_idx in layer_idx_to_clusters:
            for clsts in layer_idx_to_clusters[conv_idx]:
                num_clsts = len(clsts)
                if num_clsts > 1:
                    km = KMeans(n_clusters=1)
                    km.fit(pvalue[clsts, :])
                    sum += km.inertia_
        # print(k, sum)
        conv_idx += 1

    # print(sum)
    return sum


def generate_itr_for_model_follow_global_cluster(schedule, model):
    pca = PCA(n_components=schedule)
    result = []
    for k, v in model.state_dict().items():
        weight = v.cpu().numpy()
        if len(weight.shape) == 4:
            if len(result) == 0:  # 跳过第一个卷积层，一般不剪枝
                result.append(weight.shape[0])
            else:
                weight = np.reshape(weight, (weight.shape[0], -1))
                pca_res = pca.fit_transform(weight)
                num_channel = int(pca_res.shape[1] / 16 + 0.5) * 16
                result.append(num_channel)
                print(k, ":    ", weight.shape[0], " -> ", pca_res.shape[1], " -> ", num_channel)
    return np.array(result)


if __name__ == "__main__":
    import torchvision

    from utils.constant import RESNET50_ORIGIN_DEPS_FLATTENED
    from utils.model_utils import ModelUtils

    model = torchvision.models.resnet50(pretrained=True)
    model_utils = ModelUtils(local_rank=0)
    model_utils.register_state(model=model)
    kernel_namedvalue_list = model_utils.get_all_conv_kernel_namedvalue_as_list()
    # print(kernel_namedvalue_list)

    clusters_save_path = './clusters_save.npy'
    layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()

    print(layer_idx_to_clusters.keys())
    param_name_to_merge_matrix = generate_merge_matrix_for_kernel(deps=RESNET50_ORIGIN_DEPS_FLATTENED,
                                                                  layer_idx_to_clusters=layer_idx_to_clusters,
                                                                  kernel_namedvalue_list=kernel_namedvalue_list)
    # torch.set_printoptions(profile="full")
    # print(param_name_to_merge_matrix['layer1.0.conv1.weight'])
    print(param_name_to_merge_matrix.keys())

    # for k, v in model.state_dict().items():
    #     print(k)

    # add_vecs_to_merge_mat_dicts(param_name_to_merge_matrix)
    # print(param_name_to_merge_matrix.keys())

    # for k, v in model.state_dict().items():
    #     if k in param_name_to_merge_matrix.keys():
    #         print(k)
