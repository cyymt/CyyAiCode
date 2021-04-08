from .dataProcessing.voc0712 import VOCDetection, VOC_CLASSES
from .dataProcessing.coco import COCODetection, COCO_CLASSES
from .dataProcessing.classDataset import ClassDataset
from .augmentations.ssdAugmentations import SSDAugmentation
from .augmentations.autoAugmentations import ImageNetPolicy
from .augmentations.classAugmentations import ClassAugmentation