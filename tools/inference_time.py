import sys, os
from torchvision.models.resnet import resnet50
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import torch.nn as nn
import time

def measure_inference_time(net, input, repeat=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        model(input)
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end-start) / repeat
 
def count_params(module):
    return sum([ p.numel() for p in module.parameters() ])



if __name__=="__main__":
    device = torch.device('cuda') 
    repeat = 100

    model = resnet50(pretrained=True).eval().to(device)
    fake_input = torch.randn(16,3,224,224).to(device)
    inference_time_before_pruning = measure_inference_time(model, fake_input, repeat)
    print("before pruning: inference time=%f s, parameters=%.1fM"%(inference_time_before_pruning, count_params(model)/1e6))

