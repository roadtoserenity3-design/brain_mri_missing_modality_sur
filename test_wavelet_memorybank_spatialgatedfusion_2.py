import torch
# from debian.debtags import output
import numpy as np

from predict_wavelet_memorybank_spatialgatedfusion import AverageMeter, test_softmax_multi_masks2
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import MultiEpochsDataLoader
import wavelet_memorybank_spatialgatedfusion
from argparse import ArgumentParser
import os

patch_size = 128

def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] Total: {total:,} | Trainable: {trainable:,}")
    return total, trainable

def try_flops_fvcore(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor):
    """
    FLOPs using fvcore (preferred). Returns None if fvcore not installed.
    """
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_table
    except Exception as e:
        print(f"[FLOPs] fvcore not available: {e}")
        return None

    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, (x, mask))
        #Total FLOPs:
        total_flops = flops.total()
        print("[FLOPs] fvcore total:", f"{total_flops/1e9:.3f} GFLOPs"), f"(for input {tuple(x.shape)} + mask {tuple(mask.shape)})"

        #Optional detailed table:
        #print(flop_count_table(flops))
        return total_flops

class FlopsWrapper(torch.nn.Module):
    """
    Wrap model(x, mask) -> model(x) with a fixed mask.
    Helpful if thop can't handle bool mask input.
    """
    def __init__(self, model, fixed_mask_bool):
        super().__init__()
        self.model = model
        self.register_buffer("fixed_mask", fixed_mask_bool)

    def forward(self, x):
        out = self.model(x, self.fixed_mask)
        return out[0] if isinstance(out, (tuple, list)) else out


def try_flops_thop(model: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor):
    """
    FLOPs using thop (fallback). Returns None if thop not installed.
    """
    try:
        from thop import profile
    except Exception as e:
        print(f"[FLOPs] thop not available: {e}")
        return None

    wrapper = FlopsWrapper(model, mask)
    wrapper.eval()
    with torch.no_grad():
        macs, params = profile(wrapper, inputs=(x,), verbose=False)
        # thop returns MACs; FLOPs ~ 2*MACs (common convention)
        flops = 2.0 * macs
        print("[FLOPs] thop:", f"{flops/1e9:.3f} GFLOPs",
              "| Params (thop):", f"{params/1e6:.3f} M")
        return flops



def measure_latency_ms(model, x, mask, use_amp=False, amp_dtype=torch.float16, warmup=10, iters=30):
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                _ = model(x, mask)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    times = []
    with torch.no_grad():
        for _ in range(iters):
            starter.record()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                _ = model(x, mask)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    return float(np.mean(times))




def measure_peak_vram_mb(model, x, mask, use_amp=False, amp_dtype=torch.float16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            _ = model(x, mask)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datapath',
                        default='/BRATS2023_Training_mmFormer_npy',
                        type=str)
    parser.add_argument('--resume', default='/model_last.pth',
                        type=str)
    parser.add_argument('--savepath', default='', type=str)


    args = parser.parse_args()

    datapath = args.datapath
    resume = args.resume
    savepath = args.savepath

    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False],
             [True, False, False, False],
             [False, True, False, True], [False, True, True, False], [True, False, True, False],
             [False, False, True, True], [True, False, False, True], [True, True, False, False],
             [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
             [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair',
                 't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
                 'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
                 'flairt1cet1t2']

    test_transform = 'Compose([NumpyType((np.float32, np.int64)),])'

    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datalist/test15splits.csv')

    num_cls = 4
    dataname = 'BRATS2023'

    test_set = Brats_loadall_test_nii(transforms=test_transform, root=datapath, test_file=test_file)
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # model = UNet_try.MultiModalUNetFusion(num_cls=num_cls)
    # model = torch.nn.DataParallel(model).cuda()

    model = wavelet_memorybank_spatialgatedfusion.MultiModalUNet_WaveletMem_SpatialFuse(
        num_cls=num_cls,
        mem_size=1024,
        key_dim=128,
        temperature=0.07,
        base=8,
        update_memory=False, #IMPORTANT: no memory update at test
        lambda_hf_energy=0.0, #no training loss at test
        lambda_use=0.0      #no usage regularizer at test
    ).cuda()

    with torch.no_grad():
        dummy_x5 = torch.zeros(1, model.c5, 8, 8, 8, device="cuda")  # c5 = base*16
        model._init_val_to_maps_if_needed(dummy_x5)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if not os.path.isfile(resume):
        raise FileNotFoundError(f"Checkpoint not found: {resume}")

    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    # best_epoch = checkpoint['epoch'] + 1
    best_epoch = checkpoint.get("epoch", -1) + 1

    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()

    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, f"output{best_epoch}.txt")

    # Dummy input for per-patch compute metrics
    dummy_x = torch.zeros(
        1, 4, patch_size, patch_size, patch_size, device="cuda", dtype=torch.float32
    )
    dummy_mask_full = torch.ones(1, 4, device="cuda", dtype=torch.bool)  # 1111



    with open(output_path, "w") as f:

        # -----------------------------
        # MODEL COMPLEXITY
        # -----------------------------
        f.write("===== MODEL COMPLEXITY =====\n")

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write(f"[Params] Total: {total_p:,} | Trainable: {train_p:,}\n")
        print(f"[Params] Total: {total_p:,} | Trainable: {train_p:,}")

        # --------------------------------------------------
        # FLOPs for full-modality (1111)
        # --------------------------------------------------
        # FLOPs (1111)
        fl_1111 = try_flops_fvcore(base_model, dummy_x, dummy_mask_full)
        if fl_1111 is None:
            fl_1111 = try_flops_thop(base_model, dummy_x, dummy_mask_full)
        if fl_1111 is not None:
            f.write(f"[FLOPs] Full modality (1111): {fl_1111 / 1e9:.3f} GFLOPs per {patch_size}^3 patch\n")

        # --------------------------------------------------
        # FLOPs averaged over 15 missing-modality masks
        # --------------------------------------------------
        all_flops = []

        for m in masks:
            dm = torch.tensor([m], device="cuda", dtype=torch.bool)

            fl = try_flops_fvcore(base_model, dummy_x, dm)
            if fl is None:
                fl = try_flops_thop(base_model, dummy_x, dm)

            if fl is not None:
                all_flops.append(fl)

        if len(all_flops) > 0:
            mean_flops = float(np.mean(all_flops))
            print(f"Mean FLOPs over 15 masks: {mean_flops / 1e9:.3f} GFLOPs")

            # If writing to file:
            f.write(f"[FLOPs] Mean over 15 masks: {mean_flops / 1e9:.3f} GFLOPs per 128^3 patch\n")
        else:
            print("FLOPs could not be computed.")

        use_amp_metrics = True
        amp_dtype_metrics = torch.float16
        lat_ms = measure_latency_ms(base_model, dummy_x, dummy_mask_full, use_amp=use_amp_metrics,
                                    amp_dtype=amp_dtype_metrics)
        vram_mb = measure_peak_vram_mb(base_model, dummy_x, dummy_mask_full, use_amp=use_amp_metrics,
                                       amp_dtype=amp_dtype_metrics)

        f.write(f"[Latency] {lat_ms:.2f} ms per {patch_size}^3 patch (mask=1111)\n")
        f.write(f"[Peak VRAM] {vram_mb:.2f} MB (mask=1111)\n")

        f.write("===== END COMPLEXITY =====\n\n")

    print(f"[Saved] computational table -> {output_path}")

    os.makedirs(savepath, exist_ok=True)
    output_path = os.path.join(savepath, f"output{best_epoch}.txt")

    test_score = AverageMeter()


    with torch.inference_mode():
        modality_labels = ["FLAIR", "T1c", "T1", "T2"]  # adjust to your channel order

      
        avg_scores = test_softmax_multi_masks2(
            test_loader=test_loader,
            model=model,
            dataname=dataname,
            feature_masks=masks,
            modality_labels=modality_labels,
            save_dir=savepath,

            save_masks=True,  # ✅ this is the key: saves .nii.gz for EVERY case
            save_png=True,  # optional: turn off debug PNG
            # OR if you want PNG for all cases:
            # save_png=True,
            png_max_cases=999999,
        )


    # Append summary scores to the same output txt
    with open(output_path, "a") as f:
        f.write("\n===== SUMMARY (scenario averages) =====\n")
        for i, sc in enumerate(avg_scores):
            wt, tc, et, etpp = sc
            f.write(f"{mask_name[i]} {masks[i]} WT={wt:.4f} TC={tc:.4f} ET={et:.4f} ETpp={etpp:.4f}\n")
        f.write("Overall mean across masks: " + str(avg_scores.mean(axis=0)) + "\n")

    print("Wrote:")
    print(" -", os.path.join(savepath, "results_all_scenarios.txt"))
    print(" -", output_path)
