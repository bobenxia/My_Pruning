from collections import OrderedDict

import numpy as np
# from model.cifar.resnet import ResNet50
from utils.cluster_params import generate_itr_for_model_follow_global_cluster, generate_itr_to_target_deps_by_schedule_vector
from utils.constant import RESNET50_INTERNAL_KERNEL_IDXES, RESNET50_ORIGIN_DEPS_FLATTENED

from utils.misc import save_hdf5, copy_files
from utils.model_utils import ModelUtils


def delete_or_keep(array, idxes, axis=None):
    if len(idxes) > 0:
        return np.delete(array, idxes, axis=axis)
    else:
        return array


def parse_succeeding_strategy(layer_idx_to_clusters, succeeding_strategy):
    if succeeding_strategy is None:
        succeeding_map = None
    elif succeeding_strategy == 'simple':
        succeeding_map = {idx: (idx + 1) for idx in layer_idx_to_clusters.keys()}
    else:
        succeeding_map = succeeding_strategy
    return succeeding_map


def csgd_prune_and_save(engine: ModelUtils, layer_idx_to_clusters, save_file, succeeding_strategy, new_deps):
    result = OrderedDict()

    succeeding_map = parse_succeeding_strategy(succeeding_strategy=succeeding_strategy,
                                               layer_idx_to_clusters=layer_idx_to_clusters)

    kernel_namedvalues = engine.get_all_kernel_namedvalue_as_list()

    for layer_idx, namedvalue in enumerate(kernel_namedvalues):
        if layer_idx not in layer_idx_to_clusters:
            continue

        k_name = namedvalue.name
        k_value = namedvalue.value
        if k_name in result:  # If this kernel has been pruned because it is subsequent to another layer
            k_value = result[k_name]

        clusters = layer_idx_to_clusters[layer_idx]

        #   Prune the kernel
        idx_to_delete = []
        for clst in clusters:
            idx_to_delete += clst[1:]  # 只保留每个聚类结果中类的第一个通道，其余的删掉
        kernel_value_pruned = delete_or_keep(k_value, idx_to_delete, axis=0)
        print('cur kernel name: {}, from {} to {}'.format(k_name, k_value.shape, kernel_value_pruned.shape))
        result[k_name] = kernel_value_pruned
        # assert new_deps[layer_idx] == kernel_value_pruned.shape[0]

        # #   Prune the related vector params
        # def handle_vecs(key_first_name, key_second_name):
        #     if key_first_name == 'bn':
        #         if 'conv' in k_name:
        #             vec_name = k_name.replace(
        #                 'conv', 'bn')  # Assume the names of conv kernel and bn params follow such a pattern.
        #         elif 'downsample' in k_name:
        #             vec_name = k_name.replace('downsample.0', 'downsample.1')
        #         elif 'shortcut' in k_name:
        #             vec_name = k_name.replace('shortcut.0', 'shortcut.1')
        #     else:
        #         vec_name = k_name
        #     vec_name = vec_name.replace('weight', key_second_name)

        #     vec_value = engine.get_param_value_by_name(vec_name)
        #     if vec_value is not None:
        #         vec_value_pruned = delete_or_keep(vec_value, idx_to_delete)
        #         result[vec_name] = vec_value_pruned
        #   Prune the related vector params
        def handle_vecs(key_name):
            vec_name = k_name.replace('conv.weight', key_name)      # Assume the names of conv kernel and bn params follow such a pattern.
            vec_value = engine.get_param_value_by_name(vec_name)
            if vec_value is not None:
                vec_value_pruned = delete_or_keep(vec_value, idx_to_delete)
                result[vec_name] = vec_value_pruned

        # handle_vecs('conv', 'bias')
        # handle_vecs('bn', 'weight')
        # handle_vecs('bn', 'bias')
        # handle_vecs('bn', 'running_mean')
        # handle_vecs('bn', 'running_var')  # 到这里已经将模型中的参数按照聚类结果进行取舍操作
        handle_vecs('conv.bias')
        handle_vecs('bn.weight')
        handle_vecs('bn.bias')
        handle_vecs('bn.running_mean')
        handle_vecs('bn.running_var')


        #   Handle the succeeding kernels 下面是处理进行过取舍操作的层的下一层
        if layer_idx not in succeeding_map:
            continue

        follows = succeeding_map[layer_idx]
        print('{} follows {}'.format(follows, layer_idx))
        if type(follows) is not list:
            follows = [follows]

        for follow_idx in follows:
            follow_kernel_value = kernel_namedvalues[follow_idx].value
            follow_kernel_name = kernel_namedvalues[follow_idx].name
            if follow_kernel_name in result:
                follow_kernel_value = result[follow_kernel_name]
            print('following kernel name: ', follow_kernel_name, 'origin shape: ', follow_kernel_value.shape)

            if follow_kernel_value.ndim == 2:  # The following is a FC layer
                fc_idx_to_delete = []
                num_filters = k_value.shape[0]
                fc_neurons_per_conv_kernel = follow_kernel_value.shape[1] // num_filters
                print('{} filters, {} neurons per kernel'.format(num_filters, fc_neurons_per_conv_kernel))
                for clst in clusters:
                    if len(clst) == 1:
                        continue
                    for i in clst[1:]:
                        fc_idx_to_delete.append(
                            np.arange(i * fc_neurons_per_conv_kernel, (i + 1) * fc_neurons_per_conv_kernel))
                    to_concat = []
                    for i in clst:
                        corresponding_neurons_idx = np.arange(i * fc_neurons_per_conv_kernel,
                                                              (i + 1) * fc_neurons_per_conv_kernel)
                        to_concat.append(np.expand_dims(follow_kernel_value[:, corresponding_neurons_idx], axis=0))
                    summed = np.sum(np.concatenate(to_concat, axis=0), axis=0)
                    reserved_idx = np.arange(clst[0] * fc_neurons_per_conv_kernel,
                                             (clst[0] + 1) * fc_neurons_per_conv_kernel)
                    follow_kernel_value[:, reserved_idx] = summed
                if len(fc_idx_to_delete) > 0:
                    follow_kernel_value = delete_or_keep(follow_kernel_value,
                                                         np.concatenate(fc_idx_to_delete, axis=0),
                                                         axis=1)
                result[follow_kernel_name] = follow_kernel_value
                print('shape of pruned following kernel: ', follow_kernel_value.shape)
            elif follow_kernel_value.ndim == 4:  # The following is a conv layer 如果是下一层是卷积层，那么就将原先的聚类结果相加放在每类的第一层
                for clst in clusters:
                    selected_k_follow = follow_kernel_value[:, clst, :, :]
                    summed_k_follow = np.sum(selected_k_follow, axis=1)
                    follow_kernel_value[:, clst[0], :, :] = summed_k_follow
                follow_kernel_value = delete_or_keep(follow_kernel_value, idx_to_delete, axis=1)
                result[follow_kernel_name] = follow_kernel_value
                print('shape of pruned following kernel: ', follow_kernel_value.shape)
            else:
                raise ValueError('wrong ndim of kernel')

    key_variables = engine.state_values()
    for name, value in key_variables.items():
        if name not in result:
            result[name] = value

    result['deps'] = new_deps

    def set_value(param, value):
        param.data = torch.from_numpy(value).cuda().type(torch.cuda.FloatTensor)

    for k, v in engine.state.model.named_parameters():
        if k in result:
            print(k, v.shape, result[k].shape)
            set_value(v, result[k])
    for k, v in engine.state.model.named_buffers():
        if k in result:
            print(k, v.shape, result[k].shape)
            set_value(v, result[k])

    print('save {} values to {} after pruning'.format(len(result), save_file))
    save_hdf5(result, save_file)


if __name__ == "__main__":
    import os
    import torch
    import shutil
    from utils.misc import load_hdf5
    from utils.constant import RESNET50_succeeding_STRATEGY
    from tools.print_model_info import get_model_infor_and_print

    output_dir = "save/20220420_test/"
    # clusters_save_path = 'save/clusters_0.80.npy'
    clusters_save_path = 'save/train_and_prune/2022-04-20T23-48-01/clusters.npy'

    layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()
    print(layer_idx_to_clusters)


    from utils.stagewise_resnet import create_SRC56,create_SRC110
    from utils.constant import rc_pacesetter_dict
    from utils.builder import ConvBuilder
    from utils.constant import rc_origin_deps_flattened, rc_internal_layers,rc_succeeding_strategy
    # deps = rc_origin_deps_flattened(9)
    # convbuilder = ConvBuilder(base_config=None)
    # model = create_SRC56(deps ,convbuilder)
    # checkpoint = torch.load("save/resnet56-CSGD-part-cluster-599-round0_0.80.pth")
    # model.load_state_dict(checkpoint["state_dict"])
    
    deps = rc_origin_deps_flattened(18)
    convbuilder = ConvBuilder(base_config=None)
    model = create_SRC110(deps ,convbuilder)
    checkpoint = torch.load("save/train_and_prune/2022-04-20T23-48-01/resnet110-CSGD-train-round0.pth")
    model.load_state_dict(checkpoint["state_dict"])

    engine = ModelUtils(local_rank=0, for_eval=True)
    engine.setup_log(name='test', log_dir=output_dir, file_name='ResNet50-CSGD-test-log.txt')
    # engine.register_state(scheduler=None, model=model, optimizer=None)
    # engine.show_variables()

    # ckpt = '/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-20T16-40-49/ResNet50-CSGD-part-cluster-599-round0.pth'
    # model = torch.load(ckpt, map_location=lambda storage, loc: storage)
    engine.register_state(scheduler=None, model=model, optimizer=None)
    # engine.show_variables()


    succeeding_strategy = rc_succeeding_strategy(18)
    print(succeeding_strategy)
    schedule = 0.80 # 超参，控制 pca 拟合情况
    target_deps = generate_itr_for_model_follow_global_cluster(schedule, model)


    csgd_prune_and_save(engine=engine,
                        layer_idx_to_clusters=layer_idx_to_clusters,
                        save_file=output_dir+'prune_mode.hdf5',
                        succeeding_strategy=succeeding_strategy,
                        new_deps=target_deps)
    # copy_files(output_dir, '/tos/save_data/my_pruning_save_data/log_and_model/SGD_CAWR_test2_600epoch/0.95-2022-02-20T16-40-49')

    fake_input = torch.randn(1, 3, 32, 32)
    model_infor = get_model_infor_and_print(engine.state.model, fake_input, 0)
    print(model_infor)
