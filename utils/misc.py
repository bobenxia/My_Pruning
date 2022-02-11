import h5py
import torch
import numpy as np
import os
import shutil


def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def set_value(param, value):
    param.data = torch.from_numpy(value).cuda().type(torch.cuda.FloatTensor)


def read_hdf5(file_path):
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            if representsInt(k):
                result[int(k)] = value
            else:
                result[str(k).replace('+', '/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result


def load_from_weights_dict(model, hdf5_dict, ignore_keyword='IGNORE_KEYWORD'):
    assigned_params = 0
    for k, v in model.named_parameters():
        new_k = k.replace(ignore_keyword, '')
        if new_k in hdf5_dict:
            print('assign {} from hdf5'.format(k))
            # print(k, v.size(), hdf5_dict[k])
            set_value(v, hdf5_dict[new_k])
            assigned_params += 1
        else:
            print('param {} not found in hdf5'.format(k))
    for k, v in model.named_buffers():
        new_k = k.replace(ignore_keyword, '')
        if new_k in hdf5_dict:
            print('buffer {} from hdf5'.format(k))
            set_value(v, hdf5_dict[new_k])
            assigned_params += 1
        else:
            print('buffer {} not found in hdf5'.format(k))


def save_hdf5(numpy_dict, file_path):
    with h5py.File(file_path, 'w') as f:
        for k, v in numpy_dict.items():
            f.create_dataset(str(k).replace('/', '+'), data=v)
    print('saved {} arrays to {}'.format(len(numpy_dict), file_path))
    f.close()


def load_hdf5(model, path):
    hdf5_dict = read_hdf5(path)
    load_from_weights_dict(model, hdf5_dict)

def copy_files(source_path, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path, exist_ok=True)

    if os.path.exists(source_path):
        # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # dirs 是一个 list，内容是该文件夹中所有的目录的名字(不包括子目录)
        # files 同样是 list, 内容是该文件夹中所有的文件(不包括子目录)
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                shutil.copy(src_file, target_path)
                print(src_file)
    print('copy files finished!')

if __name__ == "__main__":
    # # test 1
    # from model.cifar.resnet import ResNet50
    # model = ResNet50(num_classes=10)

    # for k, v in model.named_parameters():
    #     print(k, v.shape)
    # for k, v in model.named_buffers():
    #     print(k)

    # hdf5_file = "save/prune_mode.hdf5"
    # load_hdf5(model, hdf5_file)

    # test 2
    source_path = "/root/code/My_Pruning/save/train_and_prune/"
    target_path = "/Tos/test/new/"
    copy_files(source_path, target_path)
