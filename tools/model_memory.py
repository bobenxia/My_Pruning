import pynvml
import torch
from torch import nn
from torch.functional import Tensor


def get_model_gpu_mem(model: nn.Module, input: Tensor, gpu_id: int = 0) -> float:
    """gpu_id: the gpu used for model inference, default is 0

    return: gpu memory used in inference(MB)
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle)

    torch.cuda.set_device(gpu_id)
    model = model.cuda()
    input = input.cuda()
    _ = model(input)

    meminfo2 = pynvml.nvmlDeviceGetMemoryInfo(handle)
    pynvml.nvmlShutdown()

    return (meminfo2.used-meminfo1.used)/1024**2


if __name__ == "__main__":
    import torchvision.models as models
    model = models.resnet34()
    data = torch.randn(1, 3, 224, 224)

    mem = get_model_gpu_mem(model, data, 5)
    print("{:<20} {:>12} MB".format('GPU Mem Used:', mem))
