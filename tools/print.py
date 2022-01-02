import torch
import torch.nn as nn
from torch.functional import Tensor

from inference_time import get_cpu_gpu_time_in_inference
from tools.macs_and_params import get_model_macs_and_params
from tools.model_memory import get_model_gpu_mem


def get_model_info_and_print(model: nn.Module, input: Tensor, gpu_id: int = 0) -> None:
    # torch.cuda.set_device(gpu_id)
    # model = model.cuda()
    # input = input.cuda()
    image_size = tuple(input.shape[1:])

    gpu_mem = get_model_gpu_mem(model, input, gpu_id)
    Macs, params = get_model_macs_and_params(model, image_size, gpu_id)
    cpu_time, gpu_time = get_cpu_gpu_time_in_inference(model, input, gpu_id)

    print('{:<30}  {:>12} GMac'.format('Computational complexity: ', Macs))
    print('{:<30}  {:>12} M'.format('Number of parameters: ', params))
    print('{:<30}  {:>12} ms'.format('Cpu time: ', cpu_time))
    print('{:<30}  {:>12} ms'.format('Gpu time: ', gpu_time))
    print("{:<30}  {:>12} M".format('GPU Mem Used:', gpu_mem))


if __name__ == "__main__":
    import torchvision.models as models
    model = models.resnet34()
    data = torch.randn(1, 3, 224, 224)
    gpu_id = 4

    get_model_info_and_print(model, data, gpu_id)
