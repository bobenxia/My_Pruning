from collections import namedtuple
import os

from utils.cluster_params import *
from utils.constant import *
from utils.logger import get_logger

NamedParamValue = namedtuple('NamedParamValue', ['name', 'value'])


class State(object):
    def __init__(self):
        self.iteration = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None

    def register(self, **kwargs):
        for k, v in kwargs.items():
            assert k in ['iteration', 'model', 'optimizer', 'scheduler']
            setattr(self, k, v)


class ModelUtils(object):
    def __init__(self, local_rank):
        self.state = State()
        self.local_rank = local_rank
        self.logger = None

    def register_state(self, **kwargs):
        self.state.register(**kwargs)

    def setup_log(self, name='train', log_dir=None, file_name=None):
        if self.local_rank == 0:
            self.logger = get_logger(
                name, log_dir, self.local_rank, filename=file_name)
        else:
            self.logger = None
        return self.logger

    def show_variables(self):
        if self.local_rank == 0:
            print('---------- show variables -------------')
            num_params = 0
            for k, v in self.state.model.state_dict().items():
                print(k, v.shape)
                num_params += v.nelement()
            print('num params: ', num_params)
            print('--------------------------------------')

    def get_all_conv_kernel_namedvalue_as_list(self):
        result = []
        for k, v in self.state.model.state_dict().items():
            if v.dim() == 4:
                result.append(NamedParamValue(name=k, value=v.cpu().numpy()))
        return result

    def __exit__(self, type, value, tb):
        torch.cuda.empty_cache()
        if type is not None:
            if self.logger is not None:
                self.logger.warning(
                    "A exception occurred during Engine initialization, "
                    "give up running process")
            return False

    def __enter__(self):
        return self

if __name__ == "__main__":
    import torchvision
    model = torchvision.models.resnet50(pretrained=True)
    # target_deps 是需要手动设置的 itr_target_deps_7060504030
    schedule = 0.75
    target_deps = generate_itr_to_target_deps_by_schedule_vector(schedule, RESNET50_ORIGIN_DEPS_FLATTENED, RESNET50_INTERNAL_KERNEL_IDXES)
    print(RESNET50_ORIGIN_DEPS_FLATTENED)
    print(target_deps)

    # pacesetter_dict 也是需要手动设置的
    pacesetter_dict ={4: 4, 3: 4, 7: 4, 10: 4, 14: 14, 13: 14, 17: 14, 20: 14, 23: 14, 27: 27, 26: 27, 30: 27, 33: 27, 36: 27, 39: 27, 42: 27, 46: 46, 45: 46, 49: 46, 52: 46}


    model_utils = ModelUtils(local_rank=0)
    model_utils.register_state(model=model)

    # 1
    # model_utils.show_variables()

    # 2
    kernel_namedvalue_list = model_utils.get_all_conv_kernel_namedvalue_as_list()
    for i in range(len(kernel_namedvalue_list)):
        print(i, kernel_namedvalue_list[i].name, kernel_namedvalue_list[i].value.shape)

    # 看是否已经存好的聚类文件
    clusters_save_path = './clusters_save.npy'
    if not os.path.exists(clusters_save_path):
        # 3
        layer_idx_to_clusters = get_layer_idx_to_clusters(kernel_namedvalue_list=kernel_namedvalue_list,
                                                        target_deps=target_deps,
                                                        pacesetter_dict=pacesetter_dict)
    
        print(layer_idx_to_clusters.keys())
        # 因为聚类比较耗时，所以直接将结果赋值出来，过程先注释掉了。用的时候将上面恢复。
        
        # 4 pacesetter_dict 是残差结构的连接层之间的关系。
        if pacesetter_dict is not None:
            for follower_idx, pacesetter_idx in pacesetter_dict.items():
                # 这里将残差的最后一层的剪枝方案直接等同于直连的剪枝方案
                if pacesetter_idx in layer_idx_to_clusters:
                    layer_idx_to_clusters[follower_idx] = layer_idx_to_clusters[pacesetter_idx]
        print(layer_idx_to_clusters.keys())
        np.save(clusters_save_path, layer_idx_to_clusters)
    else:
        layer_idx_to_clusters = np.load(clusters_save_path, allow_pickle=True).item()
        print(layer_idx_to_clusters.keys())
    
