import argparse
from tools.utils import str2bool,mkdir
import os.path as osp
from easydict import EasyDict as edict

def parse_args(**kwargs):
    cfg = kwargs
    parser = argparse.ArgumentParser(description='Training With Pytorch')
    parser.add_argument('--mode', default='train', type=str, choices=['prune','train','test'])
    parser.add_argument('--gray', default='false', type=str2bool)
    parser.add_argument('--channel', default=3, type=int)
    parser.add_argument('--num-classes', default=62, type=int)
    parser.add_argument('--cutmix', default='false', type=str2bool)
    
    parser.add_argument('--teacher_arch', default='', type=str,
                        choices=['','mobilenet_025','mobilenet_05','mobilenet_075', 'mobilenet_1', 'mobilenet_2', 
                        'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam','resnet152_cbam',
                        'ghostnet','peelenet','tsing_net','vovnet27_slim', 'vovnet39', 'vovnet57'])
    parser.add_argument('--teacher_resume', default='', type=str, metavar='PATH')
    parser.add_argument('--teacher_T', default=-1, type=int)
    parser.add_argument('--teacher_alpha', default=0.9, type=float)

    parser.add_argument('--arch', default='mobilenet_05', type=str,
                        choices=['mobilenet_025','mobilenet_05','mobilenet_075', 'mobilenet_1', 'mobilenet_2', 
                        'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam','resnet152_cbam',
                        'ghostnet','peelenet','tsing_net','vovnet27_slim', 'vovnet39', 'vovnet57'])
    parser.add_argument('--resume', default='', type=str, metavar='PATH')
    
    parser.add_argument('--prune_s', default=0.001, type=float)
    parser.add_argument('--prune_mode', default='constant', type=str, choices=['','constant','global','locality'])
    parser.add_argument('--prune_percent', default=0.5, type=float)

    parser.add_argument('--binary_model', default='normal', type=str,
                        choices=['normal','bnn','bwn','xnor','bireal'])
    parser.add_argument('--start-epoch', default=1, type=int)
    parser.add_argument('--save_epochs', default=5, type=int)
    parser.add_argument('--snapshot', default='', type=str)
    parser.add_argument('--warmup', default=-1, type=int)
    parser.add_argument('--devices-id', default='0,1', type=str)
    parser.add_argument('-j', '--workers', default=6, type=int)

    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--milestones', default='15,25,30', type=str)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('-vb', '--val-batch-size', default=512, type=int)
    parser.add_argument('--base-lr', '--learning-rate', default=0.001, type=float)
    parser.add_argument('--adjust_lr', default='normal', type=str,choices=['normal','cosine','finetune'])
    parser.add_argument('--optimizer_type', default='sgd', type=str,choices=['sgd','adam'])


    parser.add_argument('--size-average', default='mean', type=str)
    parser.add_argument('--resample-num', default=132, type=int)
    parser.add_argument('--print-freq', '-p', default=20, type=int)

    
    parser.add_argument('--data_type', default='regresion', type=str,
                        choices=["regresion","recognation","detector"])
    parser.add_argument('--loss_type', default='mesLoss', type=str,
                        choices=['mseLoss','wingloss','awingloss','labelSmooth','focalLoss','lsoftmax', \
                            'arcface', 'sphereface', 'cosface'])

    # regresion data or recognation data
    parser.add_argument('--train_file',
                        default='', type=str)
    parser.add_argument('--test_file',
                        default='', type=str)
    parser.add_argument('--root', default='')

    # detector data
    parser.add_argument('--dataset_type', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
    parser.add_argument('--dataset_root', default='data/VOCdevkit/',
                    help='Dataset root directory path')


    parser.add_argument('--log-file', default='output.log', type=str)
    parser.add_argument('--log-mode', default='w', type=str)
    args = parser.parse_args()

    # some other operations
    args.devices_id = [int(d) for d in args.devices_id.split(',')]
    args.milestones = [int(m) for m in args.milestones.split(',')]
    args.channel = 1 if args.gray else 3

    snapshot_dir = osp.split(args.snapshot)[0]
    mkdir(snapshot_dir)
    cfg.update(vars(args))

    return edict(cfg)