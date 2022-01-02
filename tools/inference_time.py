import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch.functional import Tensor
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.models.resnet import resnet50

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

### Deprecated Method ###
# def measure_inference_time(model, input, repeat=100):
#     starter, ender = torch.cuda.Event(
#         enable_timing=True), torch.cuda.Event(enable_timing=True)
#     timings = np.zeros((repeat, 1))

#     device = torch.device('cuda')
#     input = input.to(device)
#     model = model.to(device)

#     # warm up
#     for _ in range(20):
#         _ = model(input)

#     with torch.no_grad():
#         for rep in range(repeat):
#             starter.record()
#             _ = model(input)
#             ender.record()
#             # wait for gpu sync
#             torch.cuda.synchronize()
#             curr_time = starter.elapsed_time(ender)
#             timings[rep] = curr_time

#     mean_syn = np.sum(timings) / repeat
#     print(mean_syn)
#     std_syn = np.std(timings)
#     return mean_syn


def get_cpu_gpu_time_in_inference(model: nn.Module,
                                  input: Tensor,
                                  gpu_id: int = 0) -> Tuple[float, float]:
    """gpu_id: the gpu used for model inference, default is 0.

    return: cpu time(us) and gpu time(us).
    """
    torch.cuda.set_device(gpu_id)
    model = model.cuda()
    input = input.cuda()

    # warm up
    for _ in range(10):
        _ = model(input)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True,
                 record_shapes=True) as prof:
        with record_function("model_inference"):
            _ = model(input)

    key_av = prof.key_averages()
    for i in range(len(key_av)):
        if 'model_inference' in key_av[i].key:
            return key_av[i].cpu_time_total, key_av[i].cuda_time_total


if __name__ == "__main__":
    device = torch.device('cuda')
    model = resnet50(pretrained=True).eval().to(device)
    fake_input = torch.randn(1, 3, 224, 224).to(device)

    cpu_time, gpu_time = get_cpu_gpu_time_in_inference(model, fake_input)
    print('{:<20}  {:>12} us'.format('Cpu time: ', cpu_time))
    print('{:<20}  {:>12} us'.format('Gpu time: ', gpu_time))
