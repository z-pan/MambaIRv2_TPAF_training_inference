# CLAUDE.md — MambaIRv2 Denoising for TPAF Microscopy Images

## Project Overview

Train and run inference with **MambaIRv2** (CVPR 2025) as a denoising model for two-photon autofluorescence (TPAF) microscopy images of ovarian cancer tissue slides. Images are 8-bit grayscale (single channel), 512×512, originally from NADH and FAD two-channel acquisitions.

- **Training set**: 4488 clean-noisy paired images
- **Validation set**: 792 clean-noisy paired images
- **Image format**: 8-bit grayscale, 512×512
- **Repository**: https://github.com/csguoh/MambaIR (main branch contains MambaIRv2)

## Execution Platform

- **Training & Inference**: Google Colab (GPU runtime)
- **Data storage**: Google Drive
- **Deliverable**: Claude Code should generate Colab-ready `.ipynb` notebooks and config `.yml` files

---

## Data Layout on Google Drive

All data is stored in the user's Google Drive under a root folder. The expected structure after mounting Google Drive at `/content/drive/MyDrive/`:

```
/content/drive/MyDrive/MambaIRv2_TPAF/
├── train_data/
│   ├── train/
│   │   ├── clean/    # 4488 ground-truth images
│   │   └── noisy/    # 4488 corresponding noisy images
│   └── val/
│       ├── clean/    # 792 ground-truth images
│       └── noisy/    # 792 corresponding noisy images
├── checkpoints/          # training checkpoints saved here (persists across sessions)
└── results/              # inference outputs saved here
```

The user should organize their Google Drive data into this structure before running the notebooks. The exact Drive path (`MambaIRv2_TPAF`) is configurable via a variable at the top of each notebook.

**CRITICAL**: Clean and noisy filenames must match (same name, same ordering). The notebook must verify this before training.

---

## Deliverables

Claude Code should generate the following files in this repo:

1. **`train.ipynb`** — Colab notebook for environment setup + training (Phases 1–4)
2. **`inference.ipynb`** — Colab notebook for running inference on new images (Phase 6)
3. **`options/train/train_MambaIRv2_TPAF_denoise.yml`** — Training config
4. **`options/test/test_MambaIRv2_TPAF_denoise.yml`** — Test/inference config

---

## Phase 1: Environment Setup (in Colab)

### 1.1 Colab Runtime

The notebook must start with instructions to set GPU runtime:
- Runtime → Change runtime type → **T4 GPU** (free tier) or **A100/V100** (Colab Pro)
- T4 has 15GB VRAM; A100 has 40GB; V100 has 16GB

### 1.2 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.3 Configuration Variables (top of notebook)

```python
# ============================================================
# USER CONFIGURATION — Edit these paths to match your Drive
# ============================================================
DRIVE_ROOT = "/content/drive/MyDrive/MambaIRv2_TPAF"

# Data paths (on Google Drive)
TRAIN_CLEAN = f"{DRIVE_ROOT}/train_data/train/clean"
TRAIN_NOISY = f"{DRIVE_ROOT}/train_data/train/noisy"
VAL_CLEAN   = f"{DRIVE_ROOT}/train_data/val/clean"
VAL_NOISY   = f"{DRIVE_ROOT}/train_data/val/noisy"

# Output paths (on Google Drive, persists across sessions)
CHECKPOINT_DIR = f"{DRIVE_ROOT}/checkpoints"
RESULTS_DIR    = f"{DRIVE_ROOT}/results"

# Training settings
NUM_GPU = 1              # Colab = single GPU
BATCH_SIZE = 4           # 4 for T4 (15GB), 8 for A100 (40GB)
GT_SIZE = 128            # crop size during training
TOTAL_ITER = 400000      # total training iterations
VAL_FREQ = 5000          # validate every N iterations
RESUME_TRAINING = False  # set True to resume from checkpoint
# ============================================================
```

### 1.4 Clone Repo & Install Dependencies

Colab uses pip (not conda). The install sequence is critical — order matters:

```bash
# Clone MambaIR
%cd /content
!git clone https://github.com/csguoh/MambaIR.git
%cd /content/MambaIR

# Install PyTorch (Colab may already have a compatible version)
# Check existing version first; only reinstall if needed
!python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Install Mamba dependencies (CUDA must match PyTorch)
!pip install causal_conv1d==1.1.1
!pip install mamba_ssm==1.1.1

# If the above fails, try:
# !pip install causal-conv1d>=1.1.0
# !pip install mamba-ssm --no-build-isolation

# Install BasicSR and project
!pip install basicsr==1.3.5
!pip install einops timm
!pip install -e .
```

**Fallback for mamba_ssm install failure**: Colab's CUDA version may not match. In that case:
```bash
# Check CUDA version
!nvcc --version
# Install matching mamba_ssm from source
!pip install mamba-ssm --no-build-isolation
```

### 1.5 Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

from basicsr.archs.mambairv2_arch import MambaIRv2
print("MambaIRv2 architecture imported successfully.")
```

---

## Phase 2: Data Preparation (in Colab)

### 2.1 Copy Data to Colab Local Disk or Symlink from Drive

Reading directly from Google Drive is slow for training (high I/O latency). Two strategies:

**Strategy A (fast, recommended if data fits): Copy data to Colab local disk**
```python
import shutil, os

LOCAL_DATA = "/content/MambaIR/datasets/TPAF_denoise"
os.makedirs(LOCAL_DATA, exist_ok=True)

# Copy from Drive to local SSD (~5-10 min for ~5000 images)
for split, src, dst_name in [
    ("train_clean", TRAIN_CLEAN, "train_clean"),
    ("train_noisy", TRAIN_NOISY, "train_noisy"),
    ("val_clean",   VAL_CLEAN,   "val_clean"),
    ("val_noisy",   VAL_NOISY,   "val_noisy"),
]:
    dst = os.path.join(LOCAL_DATA, dst_name)
    if not os.path.exists(dst):
        print(f"Copying {split} to local disk...")
        shutil.copytree(src, dst)
        print(f"  Done. {len(os.listdir(dst))} files.")
    else:
        print(f"{split} already exists locally. {len(os.listdir(dst))} files.")
```

**Strategy B (saves disk space): Symlink from Drive**
```python
import os

LOCAL_DATA = "/content/MambaIR/datasets/TPAF_denoise"
os.makedirs(LOCAL_DATA, exist_ok=True)

for src, name in [
    (TRAIN_CLEAN, "train_clean"), (TRAIN_NOISY, "train_noisy"),
    (VAL_CLEAN,   "val_clean"),   (VAL_NOISY,   "val_noisy"),
]:
    dst = os.path.join(LOCAL_DATA, name)
    if not os.path.exists(dst):
        os.symlink(src, dst)
        print(f"Linked {name} -> {src}")
```

**Recommendation**: Use Strategy A (copy to local) for training speed. Colab free tier has ~100GB local disk, which is sufficient for ~5000 512×512 grayscale images at ~260KB each ≈ 1.4GB total.

### 2.2 Verify Data Alignment

```python
import os

train_clean = sorted(os.listdir(f"{LOCAL_DATA}/train_clean"))
train_noisy = sorted(os.listdir(f"{LOCAL_DATA}/train_noisy"))
val_clean   = sorted(os.listdir(f"{LOCAL_DATA}/val_clean"))
val_noisy   = sorted(os.listdir(f"{LOCAL_DATA}/val_noisy"))

assert len(train_clean) == len(train_noisy) == 4488, \
    f"Train count mismatch: {len(train_clean)} clean, {len(train_noisy)} noisy"
assert train_clean == train_noisy, "Train filename mismatch between clean and noisy"
assert len(val_clean) == len(val_noisy) == 792, \
    f"Val count mismatch: {len(val_clean)} clean, {len(val_noisy)} noisy"
assert val_clean == val_noisy, "Val filename mismatch between clean and noisy"
print("All data checks passed!")
print(f"Train: {len(train_clean)} pairs | Val: {len(val_clean)} pairs")
```

### 2.3 Determine Channel Count (Grayscale Handling)

```python
from PIL import Image
import numpy as np
import cv2

sample_path = os.path.join(f"{LOCAL_DATA}/train_clean", train_clean[0])

# Check PIL mode
sample_img = Image.open(sample_path)
print(f"PIL mode: {sample_img.mode}")   # 'L' = grayscale, 'RGB' = 3-channel
print(f"PIL size: {sample_img.size}")

# Check OpenCV loading (BasicSR uses cv2)
sample_cv2 = cv2.imread(sample_path, cv2.IMREAD_UNCHANGED)
print(f"OpenCV shape: {sample_cv2.shape}")

# Verify with BasicSR's own dataset loader — this is the definitive test
from basicsr.data.paired_image_dataset import PairedImageDataset
test_opt = {
    'dataroot_gt': f'{LOCAL_DATA}/train_clean',
    'dataroot_lq': f'{LOCAL_DATA}/train_noisy',
    'filename_tmpl': '{}',
    'io_backend': {'type': 'disk'},
    'gt_size': 128,
    'use_hflip': True,
    'use_rot': True,
    'phase': 'train',
    'scale': 1,
}
ds = PairedImageDataset(test_opt)
sample = ds[0]
IN_CHANS = sample['gt'].shape[0]
print(f"\nBasicSR loads images as {IN_CHANS}-channel tensors.")
print(f"GT shape: {sample['gt'].shape}, LQ shape: {sample['lq'].shape}")
print(f"\n>> Setting in_chans = {IN_CHANS} for all configs.")
```

**IMPORTANT**: BasicSR's `PairedImageDataset` uses `cv2.imread()` which may auto-convert grayscale PNG to 3-channel. If that happens, we must use `in_chans=3`. The notebook determines this automatically.

---

## Phase 3: Training Configuration

### 3.1 Determine Correct model_type

Before creating our config, inspect the existing denoising config for the correct `model_type`:

```python
import os, yaml

# Find existing denoising configs
dn_config_path = None
for f in os.listdir("/content/MambaIR/options/train/mambairv2/"):
    if "DN" in f or "dn" in f or "denois" in f.lower():
        dn_config_path = f"/content/MambaIR/options/train/mambairv2/{f}"
        break

if dn_config_path:
    print(f"Found reference config: {dn_config_path}")
    with open(dn_config_path, 'r') as f:
        ref_config = yaml.safe_load(f)
    MODEL_TYPE = ref_config.get('model_type', 'UNKNOWN')
    print(f"Reference model_type: {MODEL_TYPE}")
else:
    # Fallback: search all configs
    print("No denoising config found. Searching all configs...")
    for root, dirs, files in os.walk("/content/MambaIR/options/"):
        for f in files:
            if f.endswith('.yml'):
                path = os.path.join(root, f)
                with open(path, 'r') as fh:
                    content = yaml.safe_load(fh)
                mt = content.get('model_type', '')
                print(f"  {f}: model_type={mt}")
    MODEL_TYPE = "ImageCleanModel"  # common BasicSR denoising model type
    print(f"Using fallback model_type: {MODEL_TYPE}")
```

### 3.2 Generate Training Config YAML Programmatically

The notebook should generate the config YAML dynamically based on detected settings:

```python
import os

os.makedirs("/content/MambaIR/options/train/mambairv2", exist_ok=True)

training_config = f"""# MambaIRv2 Denoising — TPAF Microscopy (Colab)
name: MambaIRv2_TPAF_Denoise
model_type: {MODEL_TYPE}
scale: 1
num_gpu: {NUM_GPU}
manual_seed: 42

datasets:
  train:
    task: DN
    name: TPAF_train
    type: PairedImageDataset
    dataroot_gt: {LOCAL_DATA}/train_clean
    dataroot_lq: {LOCAL_DATA}/train_noisy
    filename_tmpl: '{{}}'
    io_backend:
      type: disk
    gt_size: {GT_SIZE}
    use_hflip: true
    use_rot: true
    use_shuffle: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: {BATCH_SIZE}
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    task: DN
    name: TPAF_val
    type: PairedImageDataset
    dataroot_gt: {LOCAL_DATA}/val_clean
    dataroot_lq: {LOCAL_DATA}/val_noisy
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

network_g:
  type: MambaIRv2
  upscale: 1
  in_chans: {IN_CHANS}
  img_size: {GT_SIZE}
  img_range: 1.0
  embed_dim: 180
  d_state: 8
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 16
  inner_rank: 32
  num_tokens: 64
  convffn_kernel_size: 5
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'

path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 2e-4
    weight_decay: 0
    betas: [0.9, 0.999]
  scheduler:
    type: MultiStepLR
    milestones: [100000, 200000, 300000, 350000]
    gamma: 0.5
  total_iter: {TOTAL_ITER}
  warmup_iter: -1
  pixel_opt:
    type: CharbonnierLoss
    loss_weight: 1.0
    reduction: mean

val:
  val_freq: !!float {VAL_FREQ}
  save_img: false
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

logger:
  print_freq: 200
  save_checkpoint_freq: !!float {VAL_FREQ}
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

dist_params:
  backend: nccl
"""

config_path = "/content/MambaIR/options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml"
with open(config_path, 'w') as f:
    f.write(training_config)
print(f"Training config saved to: {config_path}")
```

### 3.3 Config Notes

- **`in_chans`**: Auto-detected from Phase 2.3. Typically 1 for grayscale or 3 if BasicSR auto-converts.
- **`upscale: 1`** and **`upsampler: ''`**: Essential for denoising (no upscaling).
- **`gt_size: 128`**: Training crops 128×128 patches from 512×512. Increase to 256 on A100.
- **`img_size`**: Must equal `gt_size`.
- **`batch_size_per_gpu`**: 4 for T4 (15GB), 8 for A100 (40GB), 2 if OOM.
- **`num_worker_per_gpu: 4`**: Colab has limited CPU cores; keep this lower than local setups.
- **`embed_dim: 180`, `depths: [6,6,6,6,6,6]`**: Base model config from the paper. For faster training or if OOM, use lighter config: `embed_dim: 48`, `depths: [6,6,6,6]`, `num_heads: [4,4,4,4]`.
- **Charbonnier loss**: Recommended by MambaIRv2 paper for denoising tasks.
- **`test_y_channel: false`**: Grayscale — compute metrics on full image, not Y channel.

---

## Phase 4: Training

### 4.1 Checkpoint Persistence (Critical for Colab)

Colab sessions can disconnect at any time (free tier: ~12h max, often less). Checkpoints must be saved to Google Drive to survive disconnects.

**Redirect BasicSR experiment directory to Drive via symlink:**
```python
import os, shutil

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

exp_link = "/content/MambaIR/experiments"
if os.path.exists(exp_link) and not os.path.islink(exp_link):
    shutil.rmtree(exp_link)
if not os.path.exists(exp_link):
    os.symlink(CHECKPOINT_DIR, exp_link)
    print(f"Experiments directory linked to Drive: {CHECKPOINT_DIR}")
```

This ensures all checkpoints, logs, and training states are written directly to Google Drive and persist across sessions.

### 4.2 Start Training

```python
%cd /content/MambaIR

!python basicsr/train.py \
    -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml
```

### 4.3 Resume Training After Disconnect

When a Colab session disconnects, you lose the local environment but Drive data persists. In a new session:

1. Re-run Phase 1 cells (mount Drive, clone repo, install deps)
2. Re-run Phase 2 cells (copy/symlink data)
3. Re-run Phase 3 cells (generate config)
4. Re-link experiments to Drive (Phase 4.1)
5. Then resume:

```bash
%cd /content/MambaIR

!python basicsr/train.py \
    -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml \
    --auto_resume
```

BasicSR's `--auto_resume` will find the latest training state in the experiments directory (which points to Drive) and continue from there.

Alternatively, manually specify the resume state:
```python
import glob

state_files = sorted(glob.glob(f"{CHECKPOINT_DIR}/MambaIRv2_TPAF_Denoise/training_states/*.state"))
if state_files:
    latest_state = state_files[-1]
    print(f"Latest checkpoint: {latest_state}")
else:
    print("No checkpoint found — will start from scratch.")
```

### 4.4 Monitor Training (in Colab)

```python
# TensorBoard inline
%load_ext tensorboard
%tensorboard --logdir /content/MambaIR/experiments/MambaIRv2_TPAF_Denoise/tb_logger
```

Or check log tail:
```bash
!tail -50 /content/MambaIR/experiments/MambaIRv2_TPAF_Denoise/*.log
```

### 4.5 Expected Training Behavior

- Loss (`l_pix`) should decrease from ~0.05 to ~0.005–0.01 range.
- Validation PSNR should increase. Good denoising: 30+ dB PSNR.
- If loss plateaus early or PSNR is very low (<25 dB), check data loading and normalization.

### 4.6 Training Time Estimates

| GPU | ~Time/iter | 400k iters total | Colab sessions needed |
|-----|-----------|-----------------|----------------------|
| T4 (free) | 0.5–1.0s | 55–110 hours | 5–10 sessions (12h max each) |
| A100 (Pro) | 0.2–0.4s | 22–44 hours | 2–4 sessions (24h max) |

Consider starting with `total_iter=200000` for initial experiments, then extending.

---

## Phase 5: Testing / Inference Configuration

### 5.1 Generate Test Config

```python
import os

os.makedirs("/content/MambaIR/options/test/mambairv2", exist_ok=True)

test_config = f"""# MambaIRv2 Denoising Test — TPAF Microscopy (Colab)
name: MambaIRv2_TPAF_Denoise_Test
model_type: {MODEL_TYPE}
scale: 1
num_gpu: 1

datasets:
  test_1:
    task: DN
    name: TPAF_val
    type: PairedImageDataset
    dataroot_gt: {LOCAL_DATA}/val_clean
    dataroot_lq: {LOCAL_DATA}/val_noisy
    filename_tmpl: '{{}}'
    io_backend:
      type: disk

network_g:
  type: MambaIRv2
  upscale: 1
  in_chans: {IN_CHANS}
  img_size: 512
  img_range: 1.0
  embed_dim: 180
  d_state: 8
  depths: [6, 6, 6, 6, 6, 6]
  num_heads: [6, 6, 6, 6, 6, 6]
  window_size: 16
  inner_rank: 32
  num_tokens: 64
  convffn_kernel_size: 5
  mlp_ratio: 2
  upsampler: ''
  resi_connection: '1conv'

path:
  pretrain_network_g: {CHECKPOINT_DIR}/MambaIRv2_TPAF_Denoise/models/net_g_latest.pth

val:
  save_img: true
  suffix: ~
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false
"""

test_config_path = "/content/MambaIR/options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml"
with open(test_config_path, 'w') as f:
    f.write(test_config)
print(f"Test config saved to: {test_config_path}")
```

### 5.2 Key Differences from Training Config

- **`img_size: 512`** — Full resolution at test time (not cropped patches).
- **`save_img: true`** — Saves denoised output images.
- **No training settings** — Only network, dataset, and paths.
- **`window_size=16`** — 512 must be divisible by 16. 512/16=32 ✓

---

## Phase 6: Running Inference

### 6.1 Standard Test with Metrics (Paired Data)

```python
%cd /content/MambaIR

!python basicsr/test.py \
    -opt options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml
```

Copy results to Drive:
```python
import shutil
results_src = "/content/MambaIR/results/MambaIRv2_TPAF_Denoise_Test"
results_dst = f"{RESULTS_DIR}/test_results"
if os.path.exists(results_src):
    shutil.copytree(results_src, results_dst, dirs_exist_ok=True)
    print(f"Results copied to Drive: {results_dst}")
```

### 6.2 Inference on New Unpaired Noisy Images

Place new noisy images on Google Drive at `MambaIRv2_TPAF/new_noisy_images/`.

```python
import os, glob, torch
import numpy as np
from PIL import Image
from basicsr.archs.mambairv2_arch import MambaIRv2

def load_model(checkpoint_path, in_chans, device='cuda'):
    model = MambaIRv2(
        upscale=1, in_chans=in_chans, img_size=512, img_range=1.0,
        embed_dim=180, d_state=8,
        depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16, inner_rank=32, num_tokens=64,
        convffn_kernel_size=5, mlp_ratio=2,
        upsampler='', resi_connection='1conv',
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model

def denoise_image(model, img_path, in_chans, device='cuda'):
    img = Image.open(img_path).convert('L')
    img_np = np.array(img).astype(np.float32) / 255.0
    if in_chans == 1:
        tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    else:
        tensor = torch.from_numpy(np.stack([img_np]*3, axis=0)).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    if in_chans == 3:
        output = output[:, 0:1, :, :]
    result = output.squeeze().clamp(0, 1).cpu().numpy()
    return (result * 255.0).round().astype(np.uint8)

# --- Run inference ---
CHECKPOINT = f"{CHECKPOINT_DIR}/MambaIRv2_TPAF_Denoise/models/net_g_latest.pth"
INPUT_DIR  = f"{DRIVE_ROOT}/new_noisy_images"
OUTPUT_DIR = f"{RESULTS_DIR}/denoised_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(CHECKPOINT, IN_CHANS)
print(f"Model loaded. Processing images from: {INPUT_DIR}")

img_paths = sorted([
    p for p in glob.glob(os.path.join(INPUT_DIR, '*'))
    if p.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.bmp'))
])
print(f"Found {len(img_paths)} images.")

for i, path in enumerate(img_paths):
    fname = os.path.basename(path)
    result = denoise_image(model, path, IN_CHANS)
    Image.fromarray(result, mode='L').save(os.path.join(OUTPUT_DIR, fname))
    if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
        print(f"  [{i+1}/{len(img_paths)}] Done")

print(f"All denoised images saved to: {OUTPUT_DIR}")
```

### 6.3 Tiled Inference (if OOM on Full 512×512)

If full-image inference causes OOM (possible on T4 with Base model):

```python
def denoise_tiled(model, img_tensor, tile_size=256, overlap=32, device='cuda'):
    _, _, H, W = img_tensor.shape
    stride = tile_size - overlap
    output = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = y_end - tile_size
            x_start = x_end - tile_size
            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                tile_out = model(tile)
            output[:, :, y_start:y_end, x_start:x_end] += tile_out
            weight[:, :, y_start:y_end, x_start:x_end] += 1
    return output / weight
```

**When using tiled inference**: construct the model with `img_size=tile_size` (e.g. 256), not 512.

### 6.4 Visual Comparison in Colab

```python
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def show_comparison(noisy_path, clean_path, denoised_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    noisy = np.array(Image.open(noisy_path).convert('L'))
    clean = np.array(Image.open(clean_path).convert('L'))
    denoised = np.array(Image.open(denoised_path).convert('L'))

    axes[0].imshow(noisy, cmap='gray'); axes[0].set_title('Noisy Input')
    axes[1].imshow(denoised, cmap='gray')
    axes[1].set_title(f'Denoised\nPSNR={psnr(clean,denoised,data_range=255):.2f} SSIM={ssim(clean,denoised,data_range=255):.4f}')
    axes[2].imshow(clean, cmap='gray'); axes[2].set_title('Ground Truth')
    for ax in axes: ax.axis('off')
    plt.tight_layout(); plt.show()

# Show a few examples
val_files = sorted(os.listdir(f"{LOCAL_DATA}/val_clean"))[:5]
for fname in val_files:
    show_comparison(
        f"{LOCAL_DATA}/val_noisy/{fname}",
        f"{LOCAL_DATA}/val_clean/{fname}",
        f"/content/MambaIR/results/MambaIRv2_TPAF_Denoise_Test/visualization/TPAF_val/{fname}",
    )
```

---

## Phase 7: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `in_chans=1` shape error | BasicSR loaded 3-channel. Phase 2.3 auto-detects; use `in_chans=3` |
| `model_type` not found | Check existing config (Phase 3.1). Try `ImageCleanModel` |
| OOM during training | Reduce `BATCH_SIZE` to 2 or 1. Reduce `GT_SIZE` to 64 |
| OOM during inference | Use tiled inference (Phase 6.3) with tile_size=256 |
| `mamba_ssm` install fails | Check `!nvcc --version`, match CUDA. Try `--no-build-isolation` |
| Colab disconnects | Checkpoints on Drive survive. Re-run setup then `--auto_resume` |
| Training slow on T4 | Use lighter model: `embed_dim=48, depths=[6,6,6,6]` |
| Drive I/O bottleneck | Use Strategy A in Phase 2.1 (copy data to local disk) |
| `window_size` assertion | `img_size` must be divisible by 16 |
| Images unchanged after inference | Check loss convergence and correct checkpoint path |

### Colab-Specific Tips

- **Keep the tab active** to prevent disconnection. Use a keep-alive browser extension.
- **Colab Pro/Pro+** gives longer sessions (24h), better GPUs (A100), and more RAM.
- **If training needs many sessions**, start with `total_iter=200000`, evaluate, then extend.
- **Save your generated config YAMLs to Drive** so you don't have to regenerate them each session.

---

## Phase 8: Evaluation & Metrics

```python
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import os

def compute_metrics(clean_dir, denoised_dir):
    psnrs, ssims = [], []
    for fname in sorted(os.listdir(clean_dir)):
        if not fname.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.bmp')):
            continue
        clean = np.array(Image.open(os.path.join(clean_dir, fname)).convert('L'))
        denoised = np.array(Image.open(os.path.join(denoised_dir, fname)).convert('L'))
        psnrs.append(peak_signal_noise_ratio(clean, denoised, data_range=255))
        ssims.append(structural_similarity(clean, denoised, data_range=255))
    print(f"Images evaluated: {len(psnrs)}")
    print(f"PSNR: {np.mean(psnrs):.4f} +/- {np.std(psnrs):.4f} dB")
    print(f"SSIM: {np.mean(ssims):.4f} +/- {np.std(ssims):.4f}")
    return psnrs, ssims

psnrs, ssims = compute_metrics(
    f"{LOCAL_DATA}/val_clean",
    f"/content/MambaIR/results/MambaIRv2_TPAF_Denoise_Test/visualization/TPAF_val/"
)
```

---

## Quick Reference: Colab Session Workflow

```
Every New Session:
  1. Mount Google Drive
  2. Clone repo + install deps
  3. Copy/symlink data from Drive to local
  4. Symlink experiments dir → Drive

First Training Run:
  5. Auto-detect in_chans + model_type
  6. Generate training YAML
  7. Start training

Resume After Disconnect:
  5. Re-generate configs (same settings)
  6. !python basicsr/train.py -opt ... --auto_resume

Inference:
  5. Generate test YAML
  6. Run basicsr/test.py or custom inference script
  7. Copy results to Drive
```

---

## File Checklist

Generated by notebooks (ephemeral in Colab, re-generated each session):
- [ ] `options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml`
- [ ] `options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml`

On Google Drive (persistent across sessions):
- [ ] `MambaIRv2_TPAF/train_data/` — your image data
- [ ] `MambaIRv2_TPAF/checkpoints/` — training checkpoints & logs
- [ ] `MambaIRv2_TPAF/results/` — inference outputs

---

## Notes on Adapting from Color Denoising to Grayscale

The original MambaIRv2 denoising configs are for Gaussian **color** image denoising (3-channel, synthetic noise at known sigma levels like σ=15, 25, 50). Our task differs:

1. **Grayscale (1-channel)** — handled by auto-detected `in_chans` setting.
2. **Real noise (paired data)** — we use real clean-noisy pairs, not synthetic Gaussian noise. No noise injection is needed; noisy images are provided directly.

The Charbonnier loss and Adam optimizer (lr=2e-4, MultiStepLR) match the paper's denoising recommendations.
