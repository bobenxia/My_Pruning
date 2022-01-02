import os
import time

import numpy as np
import torch
from torchvision.models import resnet18

# import torchprof

model = resnet18(pretrained=False)
device = torch.device('cuda')
model.eval()
model.to(device)
dump_input = torch.ones(1, 3, 224, 224).to(device)

# Warn-up
for _ in range(5):
    start = time.time()
    outputs = model(dump_input)
    torch.cuda.synchronize()
    end = time.time()
    print('Time:{}ms'.format((end - start) * 1000))

with torch.autograd.profiler.profile(enabled=True,
                                     use_cuda=True,
                                     record_shapes=True,
                                     profile_memory=False,
                                     with_modules=False,
                                     use_cpu=False,
                                     use_kineto=True) as prof:
    outputs = model(dump_input)
print(prof.table())
# print(prof.key_averages())

# with torchprof.Profile(model, use_cuda=True) as prof:
#     outputs = model(dump_input)
# # print(prof.display(show_events=False))
# print(prof.display(show_events=True))
# # prof.export_chrome_trace('./resnet_profile.json')
