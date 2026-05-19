import os
import time
import logging
import torch
import torch.nn.functional as F

import numpy as np
import nibabel as nib
from kornia import x
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from utils import Parser, criterions
from utils.generate import generate_snapshot

path = os.path.dirname(__file__)

patch_size = 128

#------------------------------
# SMALL HELPERS
#------------------------------
def ensure_dir(d):
    if d is not None:
        os.makedirs(d, exist_ok=True)

def mask_to_bits(mask_bool):
    return ''.join(['1' if bool(b) else '0' for b in mask_bool])

def mask_to_name(mask_bool, modality_labels=None):
    mask_bool = [bool(x) for x in mask_bool]
    bits = mask_to_bits(mask_bool)
    if modality_labels is None or len(modality_labels) != len(mask_bool):
        return bits
    present = [modality_labels[i] for i, v in enumerate(mask_bool) if v]
    missing = [modality_labels[i] for i, v in enumerate(mask_bool) if not v]
    return f"P={'-'.join(present) if present else 'None'}__M={'-'.join(missing) if missing else 'None'}__{bits}"



def save_debug_png(x, target, pred_lbl, out_path, modality_labels=None, z=None, vmax_label=3):
    """
    3 panels:
      [1] FLAIR input
      [2] GT overlay on FLAIR (solid tumor, bg transparent)
      [3] Pred overlay on FLAIR (solid tumor, bg transparent)
    """
    x = x.detach().cpu()
    target = target.detach().cpu()
    pred_lbl = pred_lbl.detach().cpu()

    _, C, H, W, Z = x.shape
    if z is None:
        z = Z // 2

    # channel to visualize (FLAIR assumed ch0)
    ch0 = 0

    img = x[0, ch0, :, :, z].numpy()
    gt  = target[0, :, :, z].numpy()
    pr  = pred_lbl[0, :, :, z].numpy()

    # nicer grayscale scaling (avoids weird/purple-looking contrast)
    p1, p99 = np.percentile(img, (1, 99))
    if p99 <= p1:
        p1, p99 = float(img.min()), float(img.max() + 1e-6)

    # mask background -> transparent
    gt_masked = np.ma.masked_where(gt == 0, gt)
    pr_masked = np.ma.masked_where(pr == 0, pr)

    fig = plt.figure(figsize=(12, 4))

    cmap = ListedColormap([
        (0, 0, 0, 0),  # background transparent
        (1, 0, 0, 1),  # label 1 -> red
        (0, 1, 0, 1),  # label 2 -> green
        # (1, 1, 0, 1),  # label 1 -> yellow
        (0, 0, 1, 1),  # 3 -> blue

    ])

    # 1) Input only
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(img, cmap="gray", vmin=p1, vmax=p99)
    ax1.set_title(modality_labels[ch0] if modality_labels else "Input (FLAIR)")
    ax1.axis("off")

    # 2) GT overlay
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(img, cmap="gray", vmin=p1, vmax=p99)
    ax2.imshow(gt_masked, cmap=cmap, vmin=0, vmax=vmax_label,
               alpha=1.0, interpolation="nearest")
    ax2.set_title("GT (overlay)")
    ax2.axis("off")

    # 3) Pred overlay
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(img, cmap="gray", vmin=p1, vmax=p99)
    ax3.imshow(pr_masked, cmap=cmap, vmin=0, vmax=vmax_label,
               alpha=1.0, interpolation="nearest")
    ax3.set_title("Pred (overlay)")
    ax3.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def save_overlay_png_flair(x, pred_lbl, out_path, z=None, alpha=0.35):
    """
    x: (1, C, H, W, Z) float, assumes FLAIR is channel 0
    pred_lbl: (1, H, W, Z) int
    """
    x = x.detach().cpu()
    pred_lbl = pred_lbl.detach().cpu()

    _, C, H, W, Z = x.shape
    if z is None:
        z = Z // 2

    flair = x[0, 0, :, :, z].numpy()
    pr = pred_lbl[0, :, :, z].numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(flair, cmap="gray")
    plt.imshow(pr, alpha=alpha)  # default colormap is fine
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def mask_modal(x, mask):
    # B, C, H, W, Z = x.size()
    y = torch.zeros_like(x)
    y[mask, ...] = x[mask, ...]
    return y

def softmax_output_dice_class4(output, target):
    eps = 1e-8

    ##### Label 1 #####
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    ##### Label 2 #####
    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    ##### Label 3 #####
    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ##### Post Processing #####
    if torch.sum(o3) < 500:
        o4 = o3 * 0.0
    else:
        o4 = o3

    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3
    t_whole = t1 + t2 + t3
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat(
        (torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1),
         torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def softmax_output_dice_class5(output, target):
    eps = 1e-8

    ##### Label 1 #####
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    necrosis_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    non_enhancing_dice = intersect3 / denominator3

    o4 = (output == 4).float()
    t4 = (target == 4).float()
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice = intersect4 / denominator4

    ##### Post Processing
    if torch.sum(o4) < 500:
        o5 = o4 * 0
    else:
        o5 = o4

    t5 = t4

    intersect5 = torch.sum(2 * (o5 * t5), dim=(1, 2, 3)) + eps
    denominator5 = torch.sum(o5, dim=(1, 2, 3)) + torch.sum(t5, dim=(1, 2, 3)) + eps
    enhancing_dice_postpro = intersect5 / denominator5

    o_whole = o1 + o2 + o3 + o4
    t_whole = t1 + t2 + t3 + t4
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1, 2, 3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1, 2, 3)) + torch.sum(t_whole, dim=(1, 2, 3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3 + o4
    t_core = t1 + t3 + t4
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1, 2, 3)) + eps
    denominator_core = torch.sum(o_core, dim=(1, 2, 3)) + torch.sum(t_core, dim=(1, 2, 3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat(
        (torch.unsqueeze(necrosis_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(non_enhancing_dice, 1),
         torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat(
        (torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1),
         torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()

def test_softmax(
        test_loader,
        model,
        dataname='BRATS2020',
        feature_mask=None,
        compute_loss = True,
        save_masks=False,
        save_dir=None,
        index=0):

    H, W, T = 240, 240, 155
    loss = 0.0
    # model.module.is_training = False
    # works for both DataParallel and normal model
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "is_training"):
        m.is_training = False
    model.eval()
    vals_evaluation = AverageMeter()
    vals_separate = AverageMeter()

    one_tensor = torch.ones(1, 1, patch_size, patch_size, patch_size).float().cuda()

    if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'ncr_net', 'edema', 'enhancing'
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'
        class_separate = 'necrosis', 'edema', 'non_enhancing', 'enhancing'

    for i, data in enumerate(test_loader):
        target = data[1].cuda()
        x = data[0].cuda()

        #FIX: dataset gives (B, H, W, Z, C). Convert to (B, C, H, W, Z)
        if x.ndim == 5 and x.shape[-1] in (1,2,3,4):
            x = x.permute(0,4,1,2,3).contiguous()

        names = data[-1]
        yo = data[3].cuda()

        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.cuda().bool()

        # =====  Pad to at least patch size in each spatial dim =====
        # x is (B, C, H, W, Z) here
        B, C, H0, W0, Z0 = x.shape
        orig_H, orig_W, orig_Z = H0, W0, Z0

        pad_h = max(0, patch_size - H0)
        pad_w = max(0, patch_size - W0)
        pad_z = max(0, patch_size - Z0)

        if pad_h or pad_w or pad_z:
            #F.pad for 5D uses order: (Z_left, z_right, W_left, W_right, H_left, H_right)
            x = F.pad(x, (0, pad_z, 0, pad_w, 0, pad_h))

            # target is (B, H, W, Z) in your prints
            target = F.pad(target, (0, pad_z, 0, pad_w, 0, pad_h))

            # yo: in your loader it is typically (B, num_cls, H, W, Z)
            # but if it's (B, H, W, Z), this still works by checking ndim
            if yo.ndim == 5:
                yo = F.pad(yo, (0, pad_z, 0, pad_w, 0, pad_h))
            elif yo.ndim == 4:
                yo = F.pad(yo, (0, pad_z, 0, pad_w, 0, pad_h))
            else:
                raise ValueError(f"Unexpected yo.ndim={yo.ndim}, yo.shape={tuple(yo.shape)}")

            # note: mask does not need padding
        # ====================================================================

        _, _, H, W, Z = x.size()

        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = int(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * int(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = int(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * int(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = int(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * int(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().cuda()
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        ##### Evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().cuda() # (B, 4, 133, 176, 135)

        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size]
                    # pred_part = model(x_input, mask)
                    # pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part
                    out = model(x_input, mask)  # could be tensor or (tensor, aux)
                    pred_part = out[0] if isinstance(out, (tuple, list)) else out
                    pred[:, :, h:h + patch_size, w:w + patch_size, z:z + patch_size] += pred_part

        pred = pred / weight
        b = time.time()
        # pred = pred[:, :, :H, :W, :T]
        # crop back to original (pre-pad) size
        pred = pred[:, :, :orig_H, :orig_W, :orig_Z]
        target = target[:, :orig_H, :orig_W, :orig_Z]
        if yo.ndim == 5:
            yo = yo[:, :, :orig_H, :orig_W, :orig_Z]
        else:
            yo = yo[:, :orig_H, :orig_W, :orig_Z]

        #segmentation loss
        if compute_loss:
            seg_cross_loss = criterions.softmax_weighted_loss(pred, yo, num_cls=num_cls)
            seg_dice_loss = criterions.dice_loss(pred, yo, num_cls=num_cls)
            seg_loss = seg_cross_loss + seg_dice_loss
            loss += seg_loss

        pred = torch.argmax(pred, dim=1) # (B, 133, 176, 135)

        if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
            scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)
        elif dataname == 'BRATS2015':
            scores_separate, scores_evaluation = softmax_output_dice_class5(pred, target)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i + 1), len(test_loader), (k + 1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_evaluation.update(scores_evaluation[k])
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            logging.info(msg)

            case_name = str(names[k])
            out_name = case_name  # fallback name so it ALWAYS exists

            ##### Save Predicted Mask #####
            if save_masks and save_dir is not None:
                flags_bool = mask[k].bool().cpu().numpy().tolist()
                flag_str = ''.join(['1' if f else '0' for f in flags_bool])  # -> "0001"

                # case_name = names[k]
                out_name = f"{case_name}_{flag_str}.nii.gz"
                os.makedirs(save_dir, exist_ok=True)
                out_path = os.path.join(save_dir, out_name)

                # build affine with 1.0 mm isotropic spacing
                affine = np.diag([1.0, 1.0, 1.0, 1.0])

                # gather the current volume, cast to uint8 and drop the batch dimension
                pred_np = pred[k].cpu().numpy().astype(np.uint8)  # shape (H, W, T)

                nib.save(nib.Nifti1Image(pred_np, affine), out_path)

            # write scores
            case_scores = scores_evaluation[k][0:3]
            avg_score = float(np.mean(case_scores))

           

            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                txt_path = os.path.join(save_dir, f"scores_{index}.txt")
                with open(txt_path, "a") as f:
                    f.write(
                        f"{out_name} "
                        + " ".join([f"{s:.4f}" for s in case_scores]) + " "
                        + f"{avg_score:.4f}\n"
                    )

    msg = 'Average scores:'
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_evaluation.avg)])
    # msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, vals_evaluation.avg)])
    print(msg)
    if compute_loss:
        return vals_evaluation.avg, loss / (i + 1)
    else:
        return vals_evaluation.avg


def test_softmax_multi_masks(
        test_loader,
        model,
        dataname='BRATS2020',
        feature_masks=None,         # list of masks, each mask is length-4 list/bool
        modality_labels=None,       # e.g. ["T1","T1c","T2","FLAIR"]
        compute_loss=False,         # keep False for pure evaluation
        save_masks=False,
        save_dir=None,
):
    """
    Correct protocol: run the whole dataset ONCE per mask scenario.
    Memory-safe and logs readable modality names.
    """
    assert feature_masks is not None and len(feature_masks) > 0
    M = len(feature_masks)

    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "is_training"):
        m.is_training = False
    model.eval()

    if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = ('whole', 'core', 'enhancing', 'enhancing_postpro')
        dice_fn = softmax_output_dice_class4
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = ('whole', 'core', 'enhancing', 'enhancing_postpro')
        dice_fn = softmax_output_dice_class5
    else:
        raise ValueError(f"Unknown dataname={dataname}")

    one_tensor = torch.ones(1, 1, patch_size, patch_size, patch_size,
                            device="cuda", dtype=torch.float32)
    stride = int(patch_size * 0.5)

    def make_idx_list(L):
        if L <= patch_size:
            return [0]
        cnt = int(np.ceil((L - patch_size) / stride))
        idxs = [k * stride for k in range(cnt)]
        idxs.append(L - patch_size)
        return sorted(list(dict.fromkeys(idxs)))

    all_avg_scores = []

    # ==========================================================
    # ✅ OUTER LOOP: MASK SCENARIO
    # ==========================================================
    for mi, fm in enumerate(feature_masks):

        fm = [bool(x) for x in fm]

        if modality_labels is not None and len(modality_labels) == len(fm):
            present = [modality_labels[j] for j, v in enumerate(fm) if v]
            missing = [modality_labels[j] for j, v in enumerate(fm) if not v]
            scen_name = f"Present={'+'.join(present)} | Missing={'+'.join(missing)}"
        else:
            scen_name = f"mask#{mi}:{''.join(['1' if v else '0' for v in fm])}"

        print("\n" + "=" * 90)
        print(f"[Scenario {mi}] {scen_name}   raw_mask={fm}")
        print("=" * 90)

        vals_eval = AverageMeter()
        mask = torch.tensor(np.array(fm), device="cuda").bool().unsqueeze(0)  # (1,4)

        # ======================================================
        # ✅ INNER LOOP: DATASET
        # ======================================================
        for i, data in enumerate(test_loader):

            target = data[1].cuda()     # (B,H,W,Z) usually B=1
            x = data[0].cuda()          # (B,H,W,Z,C) or (B,C,H,W,Z)

            if x.ndim == 5 and x.shape[-1] in (1, 2, 3, 4):
                x = x.permute(0, 4, 1, 2, 3).contiguous()

            names = data[-1]
            case_name = str(names[0])

            B, C, H0, W0, Z0 = x.shape
            orig_H, orig_W, orig_Z = H0, W0, Z0

            pad_h = max(0, patch_size - H0)
            pad_w = max(0, patch_size - W0)
            pad_z = max(0, patch_size - Z0)

            if pad_h or pad_w or pad_z:
                x = F.pad(x, (0, pad_z, 0, pad_w, 0, pad_h))
                target_pad = F.pad(target, (0, pad_z, 0, pad_w, 0, pad_h))
            else:
                target_pad = target

            _, _, H, W, Z = x.shape
            h_idx_list = make_idx_list(H)
            w_idx_list = make_idx_list(W)
            z_idx_list = make_idx_list(Z)

            # weight (same for this case)
            weight1 = torch.zeros(1, 1, H, W, Z, device="cuda", dtype=torch.float32)
            for hh in h_idx_list:
                for ww in w_idx_list:
                    for zz in z_idx_list:
                        weight1[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size] += one_tensor
            weight = weight1.repeat(1, num_cls, 1, 1, 1)

            pred = torch.zeros(1, num_cls, H, W, Z, device="cuda", dtype=torch.float32)

            for hh in h_idx_list:
                for ww in w_idx_list:
                    for zz in z_idx_list:
                        x_input = x[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size]

                        with torch.autocast(device_type="cuda", dtype=torch.float16):
                            out = model(x_input, mask)
                            pred_part = out[0] if isinstance(out, (tuple, list)) else out

                        pred[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size] += pred_part.float()

            pred = pred / weight

            # crop back
            pred = pred[:, :, :orig_H, :orig_W, :orig_Z]
            target_rep = target_pad[:, :orig_H, :orig_W, :orig_Z]

            pred_lbl = torch.argmax(pred, dim=1)
            _, scores_eval = dice_fn(pred_lbl, target_rep)

            vals_eval.update(scores_eval[0])

            print(f"[{i+1}/{len(test_loader)}] {case_name} "
                  + ", ".join([f"{k}:{v:.4f}" for k, v in zip(class_evaluation, scores_eval[0])]))

            del pred, pred_lbl, scores_eval
            torch.cuda.empty_cache()

        avg = vals_eval.avg
        all_avg_scores.append(avg)

        print("\n>>> Scenario Average:")
        print(f"{scen_name}  ->  " + ", ".join([f"{k}:{v:.4f}" for k, v in zip(class_evaluation, avg)]))

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, f"scenario_{mi}.txt"), "w") as f:
                f.write(scen_name + "\n")
                f.write(" ".join([f"{k}:{v:.4f}" for k, v in zip(class_evaluation, avg)]) + "\n")

    return np.stack(all_avg_scores, axis=0)  # (M,4)

def test_softmax_multi_masks2(
        test_loader,
        model,
        dataname='BRATS2020',
        feature_masks=None,          # list of masks, each mask is length-4 list/bool
        modality_labels=None,        # e.g. ["T1","T1c","T2","FLAIR"]
        compute_loss=False,          # keep False for pure evaluation
        save_masks=False,            # save predicted nifti masks
        save_dir=None,
        save_png=True,               # save debug PNG
        png_max_cases=9999,             # save PNG only for first N cases per scenario
        png_slice=None               # None -> mid-slice, or int
):
    """
    Correct protocol: run the whole dataset ONCE per mask scenario.
    Also saves results to a single txt file in save_dir:
        results_all_scenarios.txt
    """
    assert feature_masks is not None and len(feature_masks) > 0
    M = len(feature_masks)

    # model flags
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "is_training"):
        m.is_training = False
    model.eval()

    # dataset-specific
    if dataname in ['BRATS2023', 'BRATS2021', 'BRATS2020', 'BRATS2018']:
        num_cls = 4
        class_evaluation = ('whole', 'core', 'enhancing', 'enhancing_postpro')
        dice_fn = softmax_output_dice_class4
        vmax_label = 3
    elif dataname == 'BRATS2015':
        num_cls = 5
        class_evaluation = ('whole', 'core', 'enhancing', 'enhancing_postpro')
        dice_fn = softmax_output_dice_class5
        vmax_label = 4
    else:
        raise ValueError(f"Unknown dataname={dataname}")

    # sliding window helpers
    one_tensor = torch.ones(1, 1, patch_size, patch_size, patch_size,
                            device="cuda", dtype=torch.float32)
    stride = int(patch_size * 0.5)

    def make_idx_list(L):
        if L <= patch_size:
            return [0]
        cnt = int(np.ceil((L - patch_size) / stride))
        idxs = [k * stride for k in range(cnt)]
        idxs.append(L - patch_size)
        return sorted(list(dict.fromkeys(idxs)))

    # outputs
    all_avg_scores = []

    # master txt
    ensure_dir(save_dir)
    master_txt = None
    if save_dir is not None:
        master_txt = os.path.join(save_dir, "results_all_scenarios.txt")
        with open(master_txt, "w") as f:
            f.write("#scenario_id\tscenario_name\tcase\tWT\tTC\tET\tETpp\tAVG3\n")

    # ==========================================================
    # ✅ OUTER LOOP: MASK SCENARIO (dataset once per mask)
    # ==========================================================
    for mi, fm in enumerate(feature_masks):
        fm = [bool(x) for x in fm]
        scen_tag = mask_to_name(fm, modality_labels)
        scen_name = scen_tag  # readable enough

        print("\n" + "=" * 100)
        print(f"[Scenario {mi}/{M - 1}] {scen_name}")
        print("=" * 100)

        vals_eval = AverageMeter()
        mask = torch.tensor(np.array(fm), device="cuda").bool().unsqueeze(0)  # (1,4)

        # scenario folders
        png_dir = None
        if save_dir is not None and save_png:
            png_dir = os.path.join(save_dir, "debug_png", f"scenario_{mi}_{scen_tag}")
            ensure_dir(png_dir)

        if save_dir is not None and save_masks:
            ensure_dir(save_dir)

        # ======================================================
        # ✅ INNER LOOP: DATASET
        # ======================================================
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                target = data[1].cuda()  # (B,H,W,Z)
                x = data[0].cuda()  # (B,H,W,Z,C) or (B,C,H,W,Z)

                if x.ndim == 5 and x.shape[-1] in (1, 2, 3, 4):
                    x = x.permute(0, 4, 1, 2, 3).contiguous()

                names = data[-1]
                case_name = str(names[0])

                B, C, H0, W0, Z0 = x.shape
                orig_H, orig_W, orig_Z = H0, W0, Z0

                # pad
                pad_h = max(0, patch_size - H0)
                pad_w = max(0, patch_size - W0)
                pad_z = max(0, patch_size - Z0)

                if pad_h or pad_w or pad_z:
                    x_pad = F.pad(x, (0, pad_z, 0, pad_w, 0, pad_h))
                    target_pad = F.pad(target, (0, pad_z, 0, pad_w, 0, pad_h))
                else:
                    x_pad = x
                    target_pad = target

                _, _, H, W, Z = x_pad.shape
                h_idx_list = make_idx_list(H)
                w_idx_list = make_idx_list(W)
                z_idx_list = make_idx_list(Z)

                # weight
                weight1 = torch.zeros(1, 1, H, W, Z, device="cuda", dtype=torch.float32)
                for hh in h_idx_list:
                    for ww in w_idx_list:
                        for zz in z_idx_list:
                            weight1[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size] += one_tensor
                weight = weight1.repeat(1, num_cls, 1, 1, 1)

                pred = torch.zeros(1, num_cls, H, W, Z, device="cuda", dtype=torch.float32)

                # sliding window infer
                for hh in h_idx_list:
                    for ww in w_idx_list:
                        for zz in z_idx_list:
                            x_input = x_pad[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size]
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = model(x_input, mask)
                                pred_part = out[0] if isinstance(out, (tuple, list)) else out
                            pred[:, :, hh:hh + patch_size, ww:ww + patch_size, zz:zz + patch_size] += pred_part.float()

                pred = pred / weight

                # crop back
                pred = pred[:, :, :orig_H, :orig_W, :orig_Z]
                target_rep = target_pad[:, :orig_H, :orig_W, :orig_Z]

                pred_lbl = torch.argmax(pred, dim=1)  # (1,H,W,Z)

                # dice
                _, scores_eval = dice_fn(pred_lbl, target_rep)
                vals_eval.update(scores_eval[0])

                wt, tc, et, etpp = [float(v) for v in scores_eval[0]]
                avg3 = float(np.mean([wt, tc, et]))

                print(f"[{i + 1}/{len(test_loader)}] {case_name} "
                      f"WT:{wt:.4f} TC:{tc:.4f} ET:{et:.4f} ETpp:{etpp:.4f}")

                # write master txt (per-case)
                if master_txt is not None:
                    with open(master_txt, "a") as f:
                        f.write(
                            f"{mi}\t{scen_name}\t{case_name}\t{wt:.4f}\t{tc:.4f}\t{et:.4f}\t{etpp:.4f}\t{avg3:.4f}\n")

               
                if save_masks and save_dir is not None:
                    case_name_safe = case_name.replace("/", "_")
                    scen_folder = os.path.join(save_dir, "pred_masks", f"scenario_{mi:02d}_{scen_tag}")
                    ensure_dir(scen_folder)

                    out_path = os.path.join(scen_folder, f"{case_name_safe}.nii.gz")
                    affine = np.diag([1.0, 1.0, 1.0, 1.0])
                    pred_np = pred_lbl[0].detach().cpu().numpy().astype(np.uint8)
                    nib.save(nib.Nifti1Image(pred_np, affine), out_path)

              
                if png_dir is not None and i < int(png_max_cases):
                    case_name_safe = case_name.replace("/", "_")
                    png_path = os.path.join(png_dir, f"{case_name_safe}_3panel.png")
                    save_debug_png(
                        x[:, :, :orig_H, :orig_W, :orig_Z],  # unpadded original input
                        target_rep,  # GT
                        pred_lbl,  # Pred
                        png_path,
                        modality_labels=modality_labels,
                        z=png_slice,
                        vmax_label=vmax_label
                    )

                del pred, pred_lbl, scores_eval, weight, weight1
                torch.cuda.empty_cache()

        avg = vals_eval.avg
        all_avg_scores.append(avg)

        # print scenario average
        print("\n>>> Scenario Average:")
        print(f"{scen_name} -> " + ", ".join([f"{k}:{v:.4f}" for k, v in zip(class_evaluation, avg)]))

        # write scenario average to master txt
        if master_txt is not None:
            wt, tc, et, etpp = [float(v) for v in avg]
            avg3 = float(np.mean([wt, tc, et]))
            with open(master_txt, "a") as f:
                f.write(
                    f"#AVG\t{mi}\t{scen_name}\tWT:{wt:.4f}\tTC:{tc:.4f}\tET:{et:.4f}\tETpp:{etpp:.4f}\tAVG3:{avg3:.4f}\n")

    return np.stack(all_avg_scores, axis=0)  # (M,4)


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
