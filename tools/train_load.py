import os.path as osp
import time
import datetime
import torch
from torch.nn import Conv2d, Linear
import torch.nn.functional as F
import numpy as np
from tools.utils import AverageMeter
import torch.nn as nn
import torchvision.datasets as datasets
from models import *
from torch.utils.data import DataLoader
from tools.utils import (load_checkpoint, detection_collate, collate_yolov4,
                         accuracy, choice_loss, cutmix_data,
                         compute_similarity, update_binary_linear,
                         update_binary_conv, change_conv_fc_name, updateBN)
from data import VOCDetection, COCODetection, ClassDataset, SSDAugmentation, ClassAugmentation


def load_data(args):
    if args.data_type.lower() == "regresion" or args.data_type.lower(
    ) == "recognation":
        if osp.exists(args.train_file) and osp.exists(args.test_file):
            train_dataset = ClassDataset(
                root=args.root,
                file_list=args.train_file,
                data_type=args.data_type.lower(),
                gray=args.gray,
                num_classes=args.num_classes,
                transform=ClassAugmentation(gray=args.gray,
                                            parse_type='train'),
            )
            val_dataset = ClassDataset(
                root=args.root,
                file_list=args.test_file,
                data_type=args.data_type.lower(),
                gray=args.gray,
                num_classes=args.num_classes,
                transform=ClassAugmentation(gray=args.gray, parse_type='val'),
            )
        else:
            train_dataset = datasets.ImageFolder(osp.join(args.root, 'train'),
                                                 transform=ClassAugmentation(
                                                     gray=args.gray,
                                                     parse_type='train'))

            val_dataset = datasets.ImageFolder(osp.join(args.root, 'val'),
                                               transform=ClassAugmentation(
                                                   gray=args.gray,
                                                   parse_type='val'))

        # drop_last = True/False ??????????????????????????????batch?????????,batch=100,?????????36?????????,??????????????????drop_last
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.workers,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=False)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.val_batch_size,
                                num_workers=args.workers,
                                shuffle=False,
                                pin_memory=True)

    elif args.data_type.lower() == "detector":
        if args.dataset_type == 'COCO':
            train_dataset = COCODetection(root=args.dataset_root,
                                          transform=None,
                                          mosaic=False)
        if args.dataset_type == 'VOC':
            train_dataset = VOCDetection(root=args.dataset_root,
                                         transform=None,
                                         mosaic=False)
        train_loader = None
        val_loader = None

    else:
        raise Exception(f"This project not support {args.data_type} type!!!")

    return train_loader, val_loader


def load_teacher_model(args, logging):
    if args.teacher_arch and args.teacher_resume and args.teacher_T > 0:
        if args.data_type.lower() == "regresion" or args.data_type.lower(
        ) == "recognation":
            model = eval(args.teacher_arch)(num_classes=args.num_classes,
                                            input_channel=args.channel,
                                            loss_type=False)
        elif args.data_type.lower() == "detector":
            pass

        if args.teacher_resume:
            load_checkpoint(args.teacher_resume, logging, model)

        model = nn.DataParallel(model,
                                device_ids=args.devices_id).cuda()  # -> GPU

        return model
    else:
        return None


def load_model(args, logging):
    if args.data_type.lower() == "regresion" or args.data_type.lower(
    ) == "recognation":
        model = eval(args.arch)(num_classes=args.num_classes,input_channel=args.channel,loss_type=True \
            if args.loss_type=='arcface' or args.loss_type=='sphereface' or args.loss_type=='cosface' else False)
        # ????????????????????????????????????
        module_name, model_feature = list(model.named_children())[-1]
        model_output_class = model_feature.weight.shape[
            0]  # weight.shape->[classs,in_feats]
        assert model_output_class == args.num_classes, "error,model output class not match with config!"
    elif args.data_type.lower() == "detector":
        pass
    if args.binary_model != "normal":
        # ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        conv_count = 0
        for name, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                conv_count += 1
                if conv_count == 1 and args.binary_model != "bwn":
                    continue
                try:
                    exec(
                        f"model.{name} = {update_binary_conv(args.binary_model,m)}"
                    )
                except:
                    name = change_conv_fc_name(name)
                    exec(
                        f"model.{name} = {update_binary_conv(args.binary_model,m)}"
                    )
            elif isinstance(m, nn.Linear):
                try:
                    exec(
                        f"model.{name} = {update_binary_linear(args.binary_model,m)}"
                    )
                except:
                    name = change_conv_fc_name(name)
                    exec(
                        f"model.{name} = {update_binary_linear(args.binary_model,m)}"
                    )
    if args.resume:
        load_checkpoint(args.resume, logging, model)

    criterion = choice_loss(args, model).cuda()
    if args.mode != "prune":
        model = nn.DataParallel(model,
                                device_ids=args.devices_id).cuda()  # -> GPU
    else:
        model = nn.DataParallel(model, device_ids=[args.devices_id[0]
                                                   ]).cuda()  # -> GPU

    if args.optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.base_lr,
                                    momentum=0.949,
                                    weight_decay=5e-4,
                                    nesterov=True)
    elif args.optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.base_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    return model, criterion, optimizer


def train(train_loader,
          model,
          criterion,
          optimizer,
          epoch,
          lr,
          args,
          logging,
          teacher_model=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    similaritys = AverageMeter()

    model.train()
    if teacher_model != None and args.teacher_T > 0:
        teacher_model.eval()
        criterion_distillation = nn.KLDivLoss()

    end = time.time()
    epoch_size = len(train_loader)
    n_top = 5 if args.num_classes >= 5 else 2
    for i, (input_data, target) in enumerate(train_loader):
        load_t0 = time.time()
        input_data = input_data.cuda(non_blocking=True)
        try:
            target = target.cuda(non_blocking=True)  # ????????????????????????????????????????????????
        except:
            target = [anno.cuda(non_blocking=True) for anno in target]

        if args.cutmix and args.data_type.lower() == "recognation":
            input_data, target_a, target_b, lam = cutmix_data(input_data,
                                                              target,
                                                              use_cuda=True)

        if args.loss_type == 'arcface' or args.loss_type == 'sphereface' or args.loss_type == 'cosface':
            output_acc, output = model(input_data)
        else:
            output = model(input_data)

        if args.cutmix and args.data_type.lower() == "recognation":
            loss = criterion(output, target_a) * lam + criterion(
                output, target_b) * (1. - lam)
        else:
            loss = criterion(output, target)

        if teacher_model != None and args.teacher_T > 0:
            with torch.no_grad():
                output_teacher = teacher_model(input_data)
            if args.loss_type == 'arcface' or args.loss_type == 'sphereface' or args.loss_type == 'cosface':
                output_S = F.log_softmax(output_acc / args.teacher_T, dim=1)
            else:
                output_S = F.log_softmax(output / args.teacher_T, dim=1)
            output_T = F.softmax(output_teacher / args.teacher_T, dim=1)
            loss_soft = criterion_distillation(
                output_S, output_T) * args.teacher_T * args.teacher_T
            loss = args.teacher_alpha * loss_soft + (1 -
                                                     args.teacher_alpha) * loss

        losses.update(loss.item(), input_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        if args.prune_s > 0 and args.prune_mode:
            updateBN(model, epoch, args)
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * epoch_size * (args.epochs - epoch))

        # training batch acc
        if args.data_type.lower() == "recognation":
            if args.loss_type == 'arcface' or args.loss_type == 'sphereface' or args.loss_type == 'cosface':
                acc1, acck = accuracy(output_acc, target, topk=(1, n_top))
            else:
                acc1, acck = accuracy(output, target, topk=(1, n_top))
            top1.update(acc1.item(), input_data.size(0))
            topk.update(acck.item(), input_data.size(0))
            # import pdb; pdb.set_trace()
            show_str = f'Epoch: [{epoch}/{args.epochs}][{i}/{len(train_loader)}] || '+ \
                         f'LR:{lr:8f} || '+ \
                         f'Top1-Acc:{top1.val:.4f}({top1.avg:.4f}) || '+ \
                         f'Top{n_top}-Acc:{topk.val:.4f}({topk.avg:.4f}) || '+ \
                         f'Loss:{losses.val:.6f}({losses.avg:.6f}) || '+ \
                         f'Batchtime: {batch_time:.4f} s || ETA: {str(datetime.timedelta(seconds=eta))}'
        elif args.data_type.lower() == "regresion":
            similarity_score = compute_similarity(output, target)
            similaritys.update(similarity_score.item(), input_data.size(0))
            show_str = f'Epoch: [{epoch}/{args.epochs}][{i}/{len(train_loader)}] || '+ \
                         f'LR:{lr:8f} || '+ \
                         f'Loss:{losses.val:.6f}({losses.avg:.6f}) || '+ \
                         f'similar_acc:{similaritys.val:.2f}%({similaritys.avg:.2f}%) || '+ \
                         f'Batchtime: {batch_time:.4f} s || ETA: {str(datetime.timedelta(seconds=eta))}'
        else:
            show_str = f'Epoch: [{epoch}/{args.epochs}][{i}/{len(train_loader)}] || '+ \
                         f'LR:{lr:8f} || '+ \
                         f'Loss:{losses.val:.6f}({losses.avg:.6f}) || '+ \
                         f'Batchtime: {batch_time:.4f} s || ETA: {str(datetime.timedelta(seconds=eta))}'

        # log
        if i % args.print_freq == 0:
            logging.info(show_str)
    if args.data_type.lower() == "recognation":
        logging.info(
            f'Epoch Info: [{epoch}/{args.epochs}] || Top1-Acc:{top1.avg:.4f} || Top{n_top}-Acc:{topk.avg:.4f} || Train Loss:{losses.avg:.6f}'
        )
    else:
        logging.info(
            f'Epoch Info: [{epoch}/{args.epochs}] || Train Loss:{losses.avg:.6f}'
        )


def validate(val_loader,
             model,
             criterion,
             epoch,
             logging,
             args,
             show_img_nums=-1):
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    similaritys = AverageMeter()

    n_top = 5 if args.num_classes >= 5 else 2

    model.eval()
    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            # compute output
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            if args.loss_type == 'arcface' or args.loss_type == 'sphereface' or args.loss_type == 'cosface':
                output_acc, output = model(input_data)
            else:
                output = model(input_data)

            loss = criterion(output, target)
            losses.update(loss.item(), input_data.size(0))
            if args.data_type.lower() == "recognation":
                if args.loss_type == 'arcface' or args.loss_type == 'sphereface' or args.loss_type == 'cosface':
                    acc1, acck = accuracy(output_acc, target, topk=(1, n_top))
                else:
                    acc1, acck = accuracy(output, target, topk=(1, n_top))
                top1.update(acc1.item(), input_data.size(0))
                topk.update(acck.item(), input_data.size(0))
            elif args.data_type.lower() == "regresion":
                similarity_score = compute_similarity(output, target)
                similaritys.update(similarity_score.item(), input_data.size(0))

        if args.data_type.lower() == "recognation":
            show_str = f'Val: [{epoch}][{len(val_loader)}] || '+ \
                     f'Top1-Acc:{top1.avg:.4f} || '+ \
                     f'Top{n_top}-Acc:{topk.avg:.4f} || '+ \
                     f'Test Loss:{losses.avg:.6f}'
        elif args.data_type.lower() == "regresion":
            show_str = f'Val: [{epoch}][{len(val_loader)}] || '+ \
                       f'Test Loss:{losses.avg:.6f} || '+ \
                       f'similar_acc:{similaritys.val:.2f}%({similaritys.avg:.2f}%)'
        else:
            show_str = f'Val: [{epoch}][{len(val_loader)}] || '+ \
                     f'Test Loss:{losses.avg:.6f}'
        logging.info(show_str)
    if show_img_nums > 0:
        import cv2
        from tools.utils import overlay_mask
        from PIL import Image
        from torchvision.transforms.functional import to_pil_image
        from thrid_project.torch_cam import CAM,GradCAM,GradCAMpp,SmoothGradCAMpp, \
            ScoreCAM,SSCAM,ISCAM,XGradCAM

        img_resize = 192
        img_col_row_num = int(show_img_nums**0.5)
        assert img_col_row_num * img_col_row_num == show_img_nums, "error:show_img_nums must be squred!"
        pre_img = ClassAugmentation(gray=args.gray, parse_type='val')

        cam_extractor = GradCAMpp(model)

        with open(args.test_file, "r") as f:
            lines = f.readlines()
        choise_indexs = np.random.choice(len(lines),
                                         size=show_img_nums,
                                         replace=False)
        show_img = np.empty(
            (img_resize * img_col_row_num, img_resize * img_col_row_num, 3),
            dtype=np.uint8)
        for index, line in enumerate(np.array(lines)[choise_indexs]):
            temp_path, label = line.strip().split(" ")
            img_path = osp.join(args.root, temp_path)
            pil_img = Image.open(img_path, mode='r').convert('RGB')
            input_data = pre_img(pil_img)[None, ...].cuda(non_blocking=True)
            scores = model(input_data)
            activation_map = cam_extractor(int(label), scores).cpu()
            heatmap = to_pil_image(activation_map, mode='F')
            overlay_img = overlay_mask(pil_img, heatmap, alpha=0.7)
            result = cv2.resize(
                np.asarray(overlay_img)[..., ::-1], (img_resize, img_resize))
            show_img[(index % img_col_row_num) *
                     img_resize:(index % img_col_row_num) * img_resize +
                     img_resize, (index // img_col_row_num) *
                     img_resize:(index // img_col_row_num) * img_resize +
                     img_resize] = result
        cv2.imshow("win", show_img)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()


def prune(model, logging, args):
    top1 = AverageMeter()
    topk = AverageMeter()
    n_top = 5 if args.num_classes >= 5 else 2

    if osp.exists(args.test_file):
        val_dataset = ClassDataset(
            root=args.root,
            file_list=args.test_file,
            data_type=args.data_type.lower(),
            gray=args.gray,
            num_classes=args.num_classes,
            transform=ClassAugmentation(gray=args.gray, parse_type='val'),
        )
    else:
        val_dataset = datasets.ImageFolder(osp.join(args.root, 'val'),
                                           transform=ClassAugmentation(
                                               gray=args.gray,
                                               parse_type='val'))
    val_loader = DataLoader(val_dataset,
                            batch_size=args.val_batch_size,
                            num_workers=args.workers,
                            shuffle=False,
                            pin_memory=True)

    # import thrid_project.torch_pruning as tp
    from tools.utils import keep_mask_sort_all_layers

    import prettytable as pt
    tb = pt.PrettyTable()
    tb.field_names = [
        "name", "src_channels", 'remain_channels', "cut_channels"
    ]

    model.eval()
    bns_mask = keep_mask_sort_all_layers(model, cut_rate=args.prune_percent)
    # strategy = tp.strategy.L1Strategy() # ???????????????????????????????????????rate,???????????????????????????
    # DG = tp.DependencyGraph()
    # example_inputs = torch.unsqueeze(next(iter(val_dataset))[0],0)
    # DG.build_dependency( model, example_inputs=example_inputs, output_transform=None)
    for index, layer_to_prune in enumerate(model.modules()):
        # BN???y=??*x+??????????????????????(weight)???0,??????????????(bias)???????????????
        if isinstance(layer_to_prune, nn.BatchNorm2d):
            mask_bn = bns_mask[index]
            tb.add_row([
                f'BatchNorm_{index}',
                len(mask_bn),
                int(mask_bn.sum().item()),
                int(len(mask_bn) - mask_bn.sum().item())
            ])
            layer_to_prune.weight.data.mul_(mask_bn)  # ?????????????????????????????????????????????0
            # layer_to_prune.bias.data.mul_(mask_bn)

            # ???????????????????????????????????????????????????
            # pruning_idxs = strategy(layer_to_prune.weight, amount=args.prune_percent) # ?????????????????????
            # plan = DG.get_pruning_plan( layer_to_prune, tp.prune_batchnorm, pruning_idxs)
            # plan.exec()
            # import pdb; pdb.set_trace()
    print(tb)
    with torch.no_grad():
        for i, (input_data, target) in enumerate(val_loader):
            target.requires_grad = False
            target = target.cuda(non_blocking=True)
            output = model(input_data)
            acc1, acck = accuracy(output, target, topk=(1, n_top))
            top1.update(acc1.item(), input_data.size(0))
            topk.update(acck.item(), input_data.size(0))
        show_str = f'Top1-Acc:{top1.avg:.4f} || '+ \
                   f'Top{n_top}-Acc:{topk.avg:.4f} || '
        logging.info(show_str)
        params = sum([param.nelement() for param in model.parameters()])
        logging.info("Model Number of Parameters: %.2fM" % (params / 1e6))
