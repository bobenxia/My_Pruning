# numpy实现一个 conv2d 运算
import numpy as np
import matplotlib.pyplot as plt
import torch
m = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

def conv2d(input, weight):
    # input: (batch, in_channel, in_height, in_width)
    # weight: (out_channel, in_channel, kernel_height, kernel_width)
    # output: (batch, out_channel, out_height, out_width)
    batch, in_channel, in_height, in_width = input.shape
    out_channel, _, kernel_height, kernel_width = weight.shape
    out_height = in_height - kernel_height + 1
    out_width = in_width - kernel_width + 1
    output = np.zeros((batch, out_channel, out_height, out_width))
    for i in range(batch):
        for j in range(out_channel):
            for k in range(out_height):
                for l in range(out_width):
                    output[i, j, k, l] = np.sum(input[i, :, k:k+kernel_height, l:l+kernel_width] * weight[j, :, :, :])
    return output


