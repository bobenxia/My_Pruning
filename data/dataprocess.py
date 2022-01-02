import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationDataProcessTrain:
    def __init__(
        self,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        hflip_prob=0.5,
        auto_augment_policy=None,
    ):
        #
        trans = [
            transforms.RandomResizedCrop(crop_size),
            transforms.RandomHorizontalFlip(hflip_prob),
            autoaugment.AutoAugment(
                policy=autoaugment.AutoAugmentPolicy(auto_augment_policy)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=mean, std=std),
        ]
        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationDataProcessEval:
    def __init__(
        self,
        crop_size,
        resize_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)
