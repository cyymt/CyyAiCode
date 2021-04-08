import os.path as osp
import torch
import torch.utils.data as data
from torch.nn.functional import one_hot
from PIL import Image
import numpy as np

def img_loader(path,is_gray=False):
    image = Image.open(path).convert("RGB")
    if is_gray:
        image = image.convert('L')
    return image

class ClassDataset(data.Dataset):
    def __init__(self, root, file_list, data_type, gray=False,num_classes=2,transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.file_list = file_list
        self.data_type = data_type
        self.num_classes = num_classes
        self.lines = self._get_lines()
        self.img_loader = img_loader
        self.gray = gray
        assert len(set(np.array(self.lines)[:,-1]))==self.num_classes,"Error,num_classes is not match."
        
    def _get_lines(self):
        target_list = []
        for line in open(self.file_list,"r"):
            data = line.strip().split(" ")
            image_path = osp.join(self.root,data[0])
            if self.data_type == "regresion":
                label = torch.tensor(list(map(float,data[1:])))
            elif self.data_type == "recognation":
                label = int(float(data[-1]))
                # label = one_hot(torch.arange(self.num_classes))[class_idx].float()
            else:
                raise Exception(f"This {self.data_type} type not support!!!")
            target_list.append([image_path,label])
        return target_list

    def __getitem__(self, index):
        path = self.lines[index][0]
        img = self.img_loader(path,self.gray)

        target = self.lines[index][-1]

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.lines)