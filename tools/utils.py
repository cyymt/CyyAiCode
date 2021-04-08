import torch
import torch.nn as nn
import torch.nn.functional as F
from tools.pytorchLoss import AngularPenaltySMLoss,FocalLoss, \
    LabelSmoothSoftmax,LargeMarginSoftmax,WingLoss,AdaptiveWingLoss

from matplotlib import cm
from PIL import Image
from pathlib import Path
import numpy as np
import random
import argparse
import math
import os
import thrid_project.torch_pruning as tp
strategy_keep = tp.strategy.L1Strategy()

COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

def strategy_keep_sort_all_layers(model,cut_rate=0.85):
    keep_idxs = {}
    bns = []
    for index,m in enumerate(model.modules()):
        if isinstance(m,nn.BatchNorm2d):
            weight_abs = m.weight.data.abs()
            keep_idxs[index] = weight_abs
            bns += weight_abs.cpu().numpy().tolist()
    thresh_val = sorted(bns)[int(len(bns)*cut_rate)]

    masks = {}
    for key,value in keep_idxs.items():
        value_mask = value.le(thresh_val).float()
        if value_mask.sum().item()==len(value):
            keep_index = value.argsort(descending=True)[:1] # 至少保留一个通道
            value_mask[keep_index] = 0.
        drop_index = value_mask.nonzero().view(-1).cpu().numpy().tolist()
        masks[key] = drop_index
    return masks

def keep_mask_sort_all_layers(model,cut_rate=0.85):
    keep_idxs = {}
    bns = []
    for index,m in enumerate(model.modules()):
        if isinstance(m,nn.BatchNorm2d):
            weight_abs = m.weight.data.abs()
            keep_idxs[index] = weight_abs
            bns += weight_abs.cpu().numpy().tolist()
    thresh_val = sorted(bns)[int(len(bns)*cut_rate)]

    masks = {}
    for key,value in keep_idxs.items():
        value_mask = value.gt(thresh_val).float()
        if value_mask.sum().item()==0:
            keep_index = value.argsort(descending=True)[:1] # 至少保留一个通道
            value_mask[keep_index] = 1.
        masks[key] = value_mask
    return masks

def keep_mask_per_layer(model,cut_rate=0.85):
    keep_idxs = {}
    # for layer_to_prune in [m for m in model.modules() if isinstance(m,nn.BatchNorm2d)]:
    for index,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            pruning_idxs = strategy_keep(m.weight, amount=cut_rate)
            mask = torch.ones_like(m.weight)
            mask[pruning_idxs] = 0
            keep_idxs[index] = mask
    return keep_idxs


def updateBN(model,epoch,args,reduce_scale=0.01):
    ls_rate = args.prune_s
    if args.prune_mode == 'global' and epoch > int(args.epochs * 0.5):
        ls_rate = args.prune_s * reduce_scale
    elif args.prune_mode == 'locality' and epoch > int(args.epochs * 0.5):
        keep_idxs = keep_mask_sort_all_layers(model,cut_rate=0.85) # 对每一层都进行裁剪
        
    for index,m in enumerate(model.modules()):
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(ls_rate * torch.sign(m.weight.data))  # L1
            if args.prune_mode == 'locality' and epoch > int(args.epochs * 0.5):
                mask = keep_idxs[index]
                m.weight.grad.data.sub_((1 - reduce_scale) * ls_rate * torch.sign(m.weight.data) * mask)  # L1
            # BN_grad_zero
            mask_weight = (m.weight.data != 0)
            mask_weight = mask_weight.float()
            m.weight.grad.data.mul_(mask_weight)
            m.bias.grad.data.mul_(mask_weight)

def change_conv_fc_name(name):
    for model_split_name in name.split("."):
        if model_split_name.isdigit():
            name = name.replace(f".{model_split_name}",f"[{model_split_name}]")
    return name
    
def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError('img and mask arguments need to be PIL.Image')

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError('alpha argument is expected to be of type float between 0 and 1')

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img

def log_init(logging, args):
    logging.basicConfig(
        # format='[%(asctime)s] [p%(process)s] [%(pathname)s:%(lineno)d] [%(levelname)s] %(message)s',
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.log_file, mode=args.log_mode),
            logging.StreamHandler()
        ]
    )
    print_args(args, logging)


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


def print_args(args, logging):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

def compute_similarity(output, target):
    return torch.cosine_similarity(output, target, dim=1).mean().abs()*100

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        # wrong_k = batch_size - correct_k
        res.append(correct_k.div_(batch_size))
    return res

def adjust_learning_rate(args, optimizer, epoch, milestones=None,lr_min=1e-5):
    """Sets the learning rate: milestone is a list/tuple"""
    assert len(milestones) >= 1,"milestones length at last as of one!!!"

    def to(epoch):
        if epoch <= args.warmup:
            return 1
        elif args.warmup < epoch <= milestones[0]:
            return 0
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return i
        return len(milestones)
    
    def to_value(epoch):
        for i in range(1, len(milestones)):
            if milestones[i - 1] < epoch <= milestones[i]:
                return milestones[i] - milestones[i - 1],milestones[i - 1]
        return args.epochs - milestones[-1],milestones[-1]

    if args.adjust_lr == "normal":
        n = to(epoch)
        lr = args.base_lr * (0.2 ** n)
    elif args.adjust_lr == "cosine":
        if epoch <= args.warmup:
            lr = args.base_lr * (np.cos(np.pi * (3 / 8) * (epoch * 1.0 / args.warmup)))
        elif args.warmup < epoch <= milestones[0]:
            lr = args.base_lr
        else:
            assert len(milestones) > 0,"milestones length is one at least!"
            if len(milestones)==1:
                T,T_start = milestones[0],milestones[0]
            else:
                # T = int((max(milestones) - min(milestones)) / (len(milestones) - 1))
                T,T_start = to_value(epoch)
            lr = lr_min + 0.5 * (args.base_lr - lr_min) * (1 + np.cos(np.pi * (epoch - T_start) * 1.0 / T))

    elif args.adjust_lr == "finetune":
        lr = np.where(epoch < int(args.epochs * 0.6),args.base_lr,args.base_lr*0.1)
    else:
        raise Exception(
            f"Adjust learning type: {args.adjust_lr} is not support!!!")

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def update_binary_conv(binary_type,m):
    if binary_type == 'bnn':
        result = str(m).replace("Conv2d","BNNConv2d")
    elif binary_type == 'bwn':
        result = str(m).replace("Conv2d","BWNConv2d")
    elif binary_type == 'xnor':
        result = str(m).replace("Conv2d","XnorConv2d")
    elif binary_type == 'bireal':
        result = str(m).replace("Conv2d","BiRealConv2d")
    else:
        raise Exception(f"This project is not support {binary_type} conv")
    return result

def update_binary_linear(binary_type,m):
    if binary_type == 'normal':
        result = str(m)
    elif binary_type == 'bnn':
        result = str(m).replace("Linear","BNNLinear")
    elif binary_type == 'bwn':
        result = str(m).replace("Linear","BWNLinear")
    elif binary_type == 'xnor':
        result = str(m).replace("Linear","XnorLinear")
    elif binary_type == 'bireal':
        result = str(m).replace("Linear","BiRealLinear")
    else:
        raise Exception(f"This project is not support {binary_type} conv")
    return result

def choice_loss(args,model):
    if args.loss_type=='mseLoss':
        loss_name = nn.MSELoss(reduction=args.size_average)
    elif args.loss_type=='wingloss':
        loss_name = WingLoss()
    elif args.loss_type=='awingloss':
        loss_name = AdaptiveWingLoss()
    elif args.loss_type=='labelSmooth':
        loss_name = LabelSmoothSoftmax(lb_smooth=0.1,reduction=args.size_average)
    elif args.loss_type=='focalLoss':
        loss_name = FocalLoss(alpha=0.25,gamma=2,reduction=args.size_average)
    elif args.loss_type=='lsoftmax':
        loss_name = LargeMarginSoftmax(lam=0.3,reduction=args.size_average)
    elif args.loss_type=='arcface' or args.loss_type=='sphereface' or args.loss_type=='cosface':
        # module_name,model_feature = list(model.named_children())[-1]
        # in_feats = model_feature.weight.shape[-1] # weight.shape->[classs,in_feats]
        loss_name = AngularPenaltySMLoss(loss_type=args.loss_type,reduction=args.size_average)
        # model.add_module(module_name,loss_name)
    else:
        raise Exception(f"This project not support loss type:{args.loss_type}!!!")
    return loss_name
        

def load_checkpoint(model_path, logging, model):
    if Path(model_path).is_file():
        logging.info(f'=> loading checkpoint {model_path}')
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        diff_list = list(set(checkpoint.keys()) - set(model.state_dict().keys()))
        if len(diff_list) == 0:
            model.load_state_dict(checkpoint)
        else:
            model_dict = model.state_dict()
            for k in diff_list:
                model_dict[k.replace("module.","")] = checkpoint[k]
            model.load_state_dict(model_dict)
    else:
        logging.info(f'=> no checkpoint found at {model_path}')


def save_checkpoint(state, logging, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info(f'Save checkpoint to {filename}')


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets

def collate_yolov4(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append([img])
        bboxes.append([box])
    images = np.concatenate(images, axis=0)
    images = images.transpose(0, 3, 1, 2)
    images = torch.from_numpy(images).div(255.0)
    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)
    return images.float(), bboxes

def detection_collate_retinaface(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix_data(x, y, alpha=1., use_cuda=True):
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha) # beta分布[0,1]
    else:
        lam = 1.

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    size = x.size()
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
