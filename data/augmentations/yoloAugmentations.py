import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


class SubtractMeans(object):
    def __init__(self, mean,std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std,dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        image = (image - self.mean) / self.std
        return image.astype(np.float32), boxes, labels

class YoloAugmentation(object):
    def __init__(self, size=300, mean=(0, 0, 0),std = (1,1,1)):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean,self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)