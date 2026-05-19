import os
import glob
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm

# ======================================================
# CONFIG
# ======================================================
RAW_ROOT = 
OUT_ROOT = 

MODALITY_SUFFIX = {
    "t1n": "t1n.nii.gz",
    "t1c": "t1c.nii.gz",
    "t2f": "t2f.nii.gz",
    "t2w": "t2w.nii.gz"
}

PNG_DIGITS = 3

os.makedirs(OUT_ROOT, exist_ok=True)

# ======================================================
# HELPERS
# ======================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_slice(x, eps=1e-8):
    x = x.astype(np.float32)
    x = np.nan_to_num(x)
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - mn) / (mx - mn)
    return (x * 255).astype(np.uint8)

def colorize_seg(label2d):
    label2d = label2d.astype(np.uint8)
    h, w = label2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    cmap = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    for k, v in cmap.items():
        rgb[label2d == k] = v

    return rgb

# ======================================================
# MAIN
# ======================================================
case_folders = sorted(glob.glob(os.path.join(RAW_ROOT, "BraTS-GLI-*")))

print(f"Found {len(case_folders)} cases")

for case_path in tqdm(case_folders):

    case_name = os.path.basename(case_path)
    case_out_dir = os.path.join(OUT_ROOT, case_name)
    ensure_dir(case_out_dir)

    # -------------------------------
    # Save Modalities
    # -------------------------------
    for mod_name, suffix in MODALITY_SUFFIX.items():

        nii_path = os.path.join(case_path, f"{case_name}-{suffix}")

        if not os.path.exists(nii_path):
            continue

        img = nib.load(nii_path)
        vol = img.get_fdata()   # (H, W, D)

        H, W, D = vol.shape

        mod_dir = os.path.join(case_out_dir, mod_name)
        ensure_dir(mod_dir)

        for z in range(D):
            slice_img = normalize_slice(vol[:, :, z])
            Image.fromarray(slice_img).save(
                os.path.join(mod_dir, f"{z:0{PNG_DIGITS}d}.png")
            )

    # -------------------------------
    # Save Ground Truth
    # -------------------------------
    seg_path = os.path.join(case_path, f"{case_name}-seg.nii.gz")

    if os.path.exists(seg_path):

        seg_img = nib.load(seg_path)
        seg = seg_img.get_fdata().astype(np.uint8)

        seg_dir = os.path.join(case_out_dir, "seg")
        ensure_dir(seg_dir)

        for z in range(seg.shape[2]):
            rgb = colorize_seg(seg[:, :, z])
            Image.fromarray(rgb).save(
                os.path.join(seg_dir, f"{z:0{PNG_DIGITS}d}.png")
            )

print("Done exporting RAW BraTS slices.")
