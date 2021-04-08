import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET
from data.dataProcessing.mosaic import Mosaic
from PIL import Image

# VOC_CLASSES = (  # always index 0
#     'aeroplane', 'bicycle', 'bird', 'boat',
#     'bottle', 'bus', 'car', 'cat', 'chair',
#     'cow', 'diningtable', 'dog', 'horse',
#     'motorbike', 'person', 'pottedplant',
#     'sheep', 'sofa', 'train', 'tvmonitor')
VOC_CLASSES = (  # always index 0
    'hand',)


class VOCAnnotationParse(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(int(float(cur_pt)))
            label_idx = self.class_to_ind[name]
            bndbox.append(int(label_idx))
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]
        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                #  image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 image_sets=[('2007', 'trainval'),],dataset_name='VOC0712',transform=None,mosaic=False):
        self.root = root
        self.image_set = image_sets
        self.parse_target = VOCAnnotationParse()
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.mosaic = mosaic
        self.transform = transform
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))
        if self.mosaic:
            self.mosaic_obj = Mosaic(self.get_truch(),self.get_class_len(),img_size=(600,600))


    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img_path = self._imgpath % img_id

        return img_path, self.parse_target(target)

    def get_truch(self):
        truth = {}
        for index in range(len(self.ids)):
            imgpath,boxidx = self.pull_item(index)
            truth[imgpath] = boxidx
        return truth

    def get_class_len(self):
        return len(VOC_CLASSES)

    def __getitem__(self,index):
        if self.mosaic:
            cv_img,out_bboxes = self.mosaic_obj(index,use_mixup=3,flip=False,letter_box=True,jitter=0.2,min_offset = 0.2)
            out_img = Image.fromarray(cv_img[:,:,::-1])
        else:
            img_path,out_bboxes = self.pull_item(index)
            out_img = Image.open(img_path)
        if self.transform is not None:
            out_img = self.transform(out_img)
        return out_img,out_bboxes

    def __len__(self):
        return len(self.ids)