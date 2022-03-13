

from unittest import result
import torchvision


def get_model_conv_deps(model):
    result = []
    for k, v in model.state_dict().items():
        weight = v.cpu().numpy()
        if len(weight.shape) == 4:
            result.append(weight.shape[0])

    return result

model = torchvision.models.resnet50(pretrained=True)
deps = get_model_conv_deps(model)

depsd = [64, 64, 64, 256, 256, 64, 64, 256, 64, 64, 256, 128, 128, 512, 512, 128, 128, 512, 128, 128, 512, 128, 128, 512,256, 256, 1024, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 256, 256, 1024, 512, 512, 2048, 2048, 512, 512, 2048, 512, 512, 2048]

assert deps == depsd