import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from data.augmentations.autoAugmentations import ImageNetPolicy


class ClassAugmentation(object):
    def __init__(self, resize=256,auto_augment=False,gray=False,parse_type='train'):
        self.resize = resize
        self.auto_augment = auto_augment
        self.gray = gray
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]) if not gray else \
            transforms.Normalize(mean=[0.5019607843137255],std=[0.5019607843137255])
        if self.auto_augment:
            if parse_type=="train":
                self.augment = transforms.Compose([
                    transforms.Resize((self.resize,self.resize),Image.BILINEAR),
                    # transforms.RandomCrop(224),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            else:
                self.augment = transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ])
        else:
            if parse_type=="train":
                self.augment = transforms.Compose([
                    transforms.Resize(self.resize,Image.BILINEAR),
                    transforms.RandomCrop(224),
                    transforms.RandomApply([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=15),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, hue=0.1, saturation=0.4),
                    ],p=0.5),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            else:
                self.augment = transforms.Compose([
                    transforms.Resize(self.resize,Image.BILINEAR),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    self.normalize,
                ])

    def __call__(self, img):
        return self.augment(img)