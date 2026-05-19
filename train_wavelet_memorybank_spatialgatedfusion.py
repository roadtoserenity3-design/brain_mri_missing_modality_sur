# coding=utf-8
import argparse
import os
import time
import logging
import numpy as np
# import wandb
import torch
import torch.optim
import sys

from utils.random_seed import setup_seed
from utils.parser import setup
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader
from torch.cuda.amp import autocast, GradScaler

# import UNet_try
import wavelet_memorybank_spatialgatedfusion
from utils import Parser, criterions

from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_val_nii, Brats_loadall_test_nii
from data.data_utils import init_fn

from predict_unet_memorybank_graph import AverageMeter, test_softmax

import torch.nn.functional as F

def loss_prefer_real_gates(gates: torch.Tensor, mask: torch.Tensor, margin: float = 0.1):
    """
    gates: (B, K, H, W, D) from aux["gates5"] or aux["gates4"]
    mask:  (B, K) bool, True=real modality present

    Encourage gate mass on REAL modalities > gate mass on RETRIEVED modalities.
    """
    # average over spatial dims -> (B, K)
    g = gates.mean(dim=(2, 3, 4))

    real = mask.float()
    real_mass = (g * real).sum(dim=1)              # (B,)
    fake_mass = (g * (1.0 - real)).sum(dim=1)      # (B,)

    return F.relu(fake_mass - real_mass + margin).mean()


def to_float(x):
    return float(x) if not torch.is_tensor(x) else float(x.item())


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--datapath', default='/data/missingmod/miccai2026/brats2023_dataset/missing_modalities/BRATS2023_Training_mmFormer_npy', type=str)
parser.add_argument('--dataname', default='BRATS2023', type=str)
parser.add_argument('--savepath', default='checkpoints_wavelet_memorybank_spatialgatedfusion_1000epochs_corrected', type=str)
parser.add_argument('--resume', default=None, type=str)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--num_epochs', default=1000, type=int)
parser.add_argument('--iter_per_epoch', default=150, type=int)
parser.add_argument('--seed', default=999, type=int)

# Wavelet/spatial-fuse losses
parser.add_argument('--lambda_pref', default=0.01, type=float)
parser.add_argument('--pref_margin', default=0.1, type=float)
parser.add_argument('--hf_warmup_epochs', default=5, type=int)   # set 0 to disable warmup


## parser arguments
args = parser.parse_args()
setup(args, 'training')
args.train_transforms = ('Compose([RandCrop3D((128,128,128)), '
                         'RandomRotation(10), '
                         'RandomIntensityChange((0.1,0.1)), '
                         'RandomFlip(0), '
                         'NumpyType((np.float32, np.int64)),])')
args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'

ckpts = args.savepath
os.makedirs(ckpts, exist_ok=True)

###modality missing mask
masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
         [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True],
         [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]

masks_torch = torch.from_numpy(np.array(masks))
mask_name = ['t2', 't1c', 't1', 'flair',
             't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
             'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
             'flairt1cet1t2']

print(masks_torch.int())

# val_check = []
val_check = list(range(10, args.num_epochs + 1, 10))

print(f"Validation checks: {val_check}")

def main():
    ########## setting seed
    setup_seed(args.seed)

    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25 - len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    ########## setting models
    if args.dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
    elif args.dataname == 'BRATS2015':
        num_cls = 5
    else:
        print('dataset is error')
        exit(0)

    model = wavelet_memorybank_spatialgatedfusion.MultiModalUNet_WaveletMem_SpatialFuse(
        num_cls=num_cls,
        mem_size=1024,
        key_dim=128,
        temperature=0.07,
        base=8,
        update_memory=True,
        lambda_hf_energy=0.1,
        lambda_use=0.0,
        tau_use=0.10,
        dropout=0.0,
    ).cuda()

    print(model)

    ########## Setting learning scheduler and optimizer ##########
    lr_schedule = LR_Scheduler(args.lr, args.num_epochs)
    train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999), eps=1e-08, amsgrad=True)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    ########## Setting data
    if args.dataname in ['BRATS2023', 'BRATS2020', 'BRATS2015']:
        train_file = 'datalist/train.txt'
        test_file = 'datalist/test15splits.csv'
        val_file = 'datalist/val15splits.csv'
    elif args.dataname == 'BRATS2018':
        ##### BRATS2018 contains three splits (1,2,3)
        test_file = 'datalist/Brats18_test15splits.csv'
        val_file = 'datalist/Brats18_val15splits.csv'
        train_file = 'datalist/train3.txt'

    logging.info(str(args))
    train_set = Brats_loadall_nii(transforms=args.train_transforms,
                                  root=args.datapath,
                                  num_cls=num_cls,
                                  train_file=train_file)

    val_set = Brats_loadall_val_nii(transforms=args.test_transforms,
                                    root=args.datapath,
                                    num_cls=num_cls,
                                    val_file=val_file)

    test_set = Brats_loadall_test_nii(transforms=args.test_transforms,
                                      root=args.datapath,
                                      test_file=test_file)

    train_loader = MultiEpochsDataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        worker_init_fn=init_fn,
    )

    val_loader = MultiEpochsDataLoader(
        dataset=val_set,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    test_loader = MultiEpochsDataLoader(
        dataset=test_set,
        batch_size=1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    ########## Training
    start = time.time()
    torch.set_grad_enabled(True)
    logging.info('########## TRAINING ##########')
    iter_per_epoch = len(train_loader)  # number of batches
    train_iter = iter(train_loader)
    val_Dice_best = -999999
    start_epoch = 0

    ########### Resume Training
    if args.resume is not None:
        checkpoint = torch.load(args.resume)
        logging.info('best_epoch: {}'.format(checkpoint['epoch']))

        # ---- init lazy val_to_map modules BEFORE loading weights ----
        # x5 size = crop/16 = 128/16 = 8  (since 4 MaxPool3d(2))
        with torch.no_grad():
            dummy_x5 = torch.zeros(1, model.c5, 8, 8, 8).cuda()
            model._init_val_to_maps_if_needed(dummy_x5)

        model.load_state_dict(checkpoint['state_dict'])
        val_Dice_best = checkpoint['val_Dice_best']
        optimizer.load_state_dict(checkpoint['optim_dict'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, args.num_epochs):
        step_lr = lr_schedule(optimizer, epoch)
        b = time.time()
        model.train()

        # Warmup HF loss so memory has time to fill
        if args.hf_warmup_epochs > 0 and epoch < args.hf_warmup_epochs:
            model.lambda_hf_energy = 0.0
        else:
            model.lambda_hf_energy = 0.1

        loss_epoch = 0.0

        ########## Training Epoch
        for i in range(iter_per_epoch):
            ##### Data Load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(train_loader)
                data = next(train_iter)

            x, target, mask = data[:3]  # x=(B, M=4, 128, 128, 128), target=(B, C, 128, 128, 128), mask = (B, 4)
            x = x.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True).bool()

            optimizer.zero_grad(set_to_none=True)


            with autocast(enabled=torch.cuda.is_available()):
                preds, aux = model(x, mask, return_aux=True)

                ce = criterions.softmax_weighted_loss(preds, target, num_cls=num_cls)
                dice = criterions.dice_loss(preds, target, num_cls=num_cls)
                loss_seg = ce + dice

                # loss_pref = loss_prefer_real_gates(aux["gates5"], mask, margin=args.pref_margin)

                pref_terms = []

                if "gates5" in aux:
                    pref_terms.append(loss_prefer_real_gates(aux["gates5"], mask, margin=args.pref_margin))

                if "gates4" in aux:
                    pref_terms.append(loss_prefer_real_gates(aux["gates4"], mask, margin=args.pref_margin))

                # average to keep scale stable
                loss_pref = sum(pref_terms) / max(len(pref_terms), 1)

                loss_hf = aux["hf_energy_loss"] * float(model.lambda_hf_energy)
                loss_use = aux.get("use_loss", x.new_zeros(())) * float(getattr(model, "lambda_use", 0.0))

                loss = loss_seg + (args.lambda_pref * loss_pref) + loss_hf + loss_use

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_epoch += loss.item()

            msg = (
                f"Epoch {epoch + 1}/{args.num_epochs}, Iter {i + 1}/{iter_per_epoch}, "
                f"Loss {loss.item():.4f}, Seg {loss_seg.item():.4f}, "
                f"CE {ce.item():.4f}, Dice {dice.item():.4f}, "
                f"Pref {loss_pref.item():.4f}, HF {to_float(loss_hf):.4f}, Use {to_float(loss_use):.4f}, "
                f"lr {step_lr:.2e}"
            )
            logging.info(msg)

        logging.info('train time per epoch: {}'.format(time.time() - b))

        # save every 100 epochs (100, 200, 300, ...)
        if (epoch + 1) % 100 == 0:
            file_name = os.path.join(ckpts, f'epoch_{epoch + 1:04d}.pth')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'val_Dice_best': val_Dice_best,
            }, file_name)

        ########## Save Model
        file_name = os.path.join(ckpts, 'model_last.pth')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'val_Dice_best': val_Dice_best,
        },
            file_name)

        ########## Validation and Test
        if epoch + 1 in val_check:
            print('Validate...')
            with torch.no_grad():
                dice_score, seg_loss = test_softmax(
                    val_loader,
                    model,
                    dataname=args.dataname,
                    save_dir=args.savepath,
                )

            logging.info(
                f"Validate epoch={epoch}, WT={to_float(val_WT):.2f}, TC={to_float(val_TC):.2f}, "
                f"ET={to_float(val_ET):.2f}, ETpp={to_float(val_ETpp):.2f}, loss={float(seg_loss):.2f}"
            )

            val_dice = (val_ET + val_WT + val_TC) / 3

            if val_dice > val_Dice_best:
                val_Dice_best = val_dice.item()
                print('save best model ...')
                file_name = os.path.join(ckpts, 'best.pth')
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optim_dict': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'val_Dice_best': val_Dice_best,
                },
                    file_name)

                print('testing ...')
                test_score = AverageMeter()

                with torch.no_grad():
                    dice_score, seg_loss = test_softmax(
                        test_loader,
                        model, dataname=args.dataname)
                test_WT, test_TC, test_ET, test_ETpp = dice_score
                logging.info('Testing epoch = {}, WT = {:.2}, TC = {:.2}, ET = {:.2}, ET_postpro = {:.2}'.format(epoch,
                                                                                                                 test_WT.item(),
                                                                                                                 test_TC.item(),
                                                                                                                 test_ET.item(),
                                                                                                                 test_ETpp.item()))

                test_dice = (test_ET + test_WT + test_TC) / 3


                model.train()

    msg = 'total time: {:.4f} hours'.format((time.time() - start) / 3600)
    logging.info(msg)

def to_float(x):
    return float(x) if not torch.is_tensor(x) else float(x.item())

if __name__ == '__main__':
    main()





