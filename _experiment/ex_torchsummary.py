import pynvml
import torch
import torchvision.models as models

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle)

model = models.resnet34().cuda()
data = torch.randn(1, 3, 224, 224).cuda()
_ = model(data)

meminfo2 = pynvml.nvmlDeviceGetMemoryInfo(handle)
print((meminfo2.used-meminfo1.used)/1024**2)  # 已用显存大小
