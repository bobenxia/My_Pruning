from typing import Tuple

import torch
from ptflops import get_model_complexity_info
from torch import nn


def get_model_macs_and_params(model: nn.Module,
                              image_size: tuple,
                              gpu_id: int = 0) -> Tuple[float, float]:
    """
    image_size: does not include batch_size, just for example, (3,224,224);
    gpu_id: the gpu used for model inference, default is 0.

    return: Macs and params in model
    """
    torch.cuda.set_device(gpu_id)
    model = model.cuda()

    macs, params = get_model_complexity_info(model,
                                             image_size,
                                             as_strings=False,
                                             print_per_layer_stat=False,
                                             verbose=False)
    G_macs = round(macs / 10.**9, 3)
    M_params = round(params / 10.**6, 3)
    return G_macs, M_params


if __name__ == "__main__":
    import torchvision.models as models
    net = models.resnet18()
    macs, params = get_model_macs_and_params(net, (3, 224, 224))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
