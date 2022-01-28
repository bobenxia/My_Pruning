import h5py
import numpy as np

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
        
def read_hdf5(file_path):
    result = {}
    with h5py.File(file_path, 'r') as f:
        for k in f.keys():
            value = np.asarray(f[k])
            if representsInt(k):
                result[int(k)] = value
            else:
                result[str(k).replace('+','/')] = value
    print('read {} arrays from {}'.format(len(result), file_path))
    f.close()
    return result

def save_hdf5(numpy_dict, file_path):
    with h5py.File(file_path, 'w') as f:
        for k,v in numpy_dict.items():
            f.create_dataset(str(k).replace('/','+'), data=v)
    print('saved {} arrays to {}'.format(len(numpy_dict), file_path))
    f.close()