from pytorch_resnet_cifar10 import resnet
import torch

model = resnet.resnet56()
path = "./pytorch_resnet_cifar10/pretrained_models/resnet56-4bfd9763.th"
checkpoint = torch.load(path)
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
    name = k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

print(new_state_dict.keys())