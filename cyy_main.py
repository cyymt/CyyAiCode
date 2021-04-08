#!/usr/bin/env python3
# coding: utf-8
import logging
import torch
from pathlib import Path
import torch.backends.cudnn as cudnn

from tools.utils import log_init,save_checkpoint,adjust_learning_rate
from tools.args_config import parse_args
from tools.train_load import train,validate,load_model,load_data,load_teacher_model,prune
from tools.cfg import Cfg
import warnings
warnings.filterwarnings("ignore", category=Warning)


def main():
    args = parse_args(**Cfg)

    log_init(logging,args)
    assert torch.cuda.is_available(), 'no cuda'

    torch.cuda.set_device(args.devices_id[0])  # fix bug for `ERROR: all tensors must be on devices[0]`

    # step1: load model,criterion,optimizer
    model,criterion,optimizer = load_model(args,logging)
    teacher_model = load_teacher_model(args,logging)

    # step2: load data
    train_loader,val_loader = load_data(args)
    
    # step3: run
    cudnn.benchmark = True
    if args.mode == "train":
        for epoch in range(args.start_epoch, args.epochs + 1):
            # adjust learning rate
            lr = adjust_learning_rate(args,optimizer,epoch,args.milestones)
            
            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch,lr,args,logging,teacher_model=teacher_model)
            if epoch % args.save_epochs==0 or epoch==args.epochs:
                filename = f'{args.snapshot}_checkpoint_epoch_{epoch}.pth.tar'
                save_checkpoint(
                    model.state_dict(),
                    logging,
                    filename
                )
                if val_loader != None:
                    validate(val_loader, model, criterion, epoch,logging,args)
    elif args.mode=='test' and val_loader != None and Path(args.resume).is_file():
        logging.info('Testing model acc......')
        validate(val_loader, model, criterion, args.start_epoch,logging,args,show_img_nums=25)
        # 计算模型总参数量
        params = sum([param.nelement() for param in model.parameters()])
        logging.info("Model Number of Parameters: %.2fM"%(params/1e6))
    elif args.mode=='prune' and args.prune_percent>0 and Path(args.resume).is_file():
        logging.info('Pruning model......')
        prune(model,logging,args)
    else:
        raise Exception("Please check your config file!")

if __name__ == '__main__':
    main()
