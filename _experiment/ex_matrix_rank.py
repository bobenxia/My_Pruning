import numpy as np
import torch
from numpy.linalg import matrix_rank


if __name__ == "__main__":
    torch.manual_seed(3)

    # import torchvision
    # model = torchvision.models.resnet50(pretrained=True)
    ckpt = "save/train_and_prune/2022-02-20T16-40-49/ResNet50-CSGD-global-cluster-199-round0.pth"
    print("Load model from %s" % (ckpt))
    # need load model to cpu, avoid computing model GPU memory errors
    model = torch.load(ckpt, map_location=torch.device("cpu"))


    for k, v in model.state_dict().items():
        weight = v.cpu().numpy()
        if len(weight.shape) == 4:
            weight = np.reshape(weight, (weight.shape[0], -1))
            rank = matrix_rank(weight)
            print(rank)