import torchvision
import torch
import math

model = torchvision.models.resnet50(pretrained=True)
for k, v in model.state_dict().items():
    if len(v.shape) == 4 :
        print(k, v.shape)
        torch.nn.init.xavier_normal_(v.data)
    elif len(v.shape) == 1:
        torch.nn.init.normal_(v.data)

# for k, v in model.named_modules():
#     if isinstance(v, torch.nn.Conv2d):
#         print(k, v.weight.shape)
#         print(k, v.bias)
#         # print(v.weight)
#         # torch.nn.init.kaiming_uniform_(v.weight.data, a=math.sqrt(5))
#         # print(v.weight)
#         # break
#     # elif isinstance(v, torch.nn.b)