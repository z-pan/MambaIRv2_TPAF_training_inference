# CLAUDE.md — MambaIRv2 Denoising for TPAF Microscopy Images

## Project Overview

Train and run inference with **MambaIRv2** (CVPR 2025) as a denoising model for two-photon autofluorescence (TPAF) microscopy images of ovarian cancer tissue slides. Images are 8-bit grayscale (single channel), 512×512, originally from NADH and FAD two-channel acquisitions.

- **Training set**: 4488 clean-noisy paired images
- **Validation set**: 792 clean-noisy paired images
- **Image format**: 8-bit grayscale, 512×512
- **Repository**: https://github.com/csguoh/MambaIR (main branch contains MambaIRv2)

---

## Data Layout

```
train_data/
├── train/
│   ├── clean/    # 4488 ground-truth images
│   └── noisy/    # 4488 corresponding noisy images
└── val/
    ├── clean/    # 792 ground-truth images
    └── noisy/    # 792 corresponding noisy images
```

**CRITICAL**: Clean and noisy filenames must match (same name, same ordering). Verify this before training.

---

## Phase 1: Environment Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/csguoh/MambaIR.git
cd MambaIR
```

### 1.2 Create Conda Environment

The repo was tested with Python 3.10, PyTorch 2.0.1, CUDA 11.8. Create environment:

```bash
conda create -n mambairv2 python=3.10 -y
conda activate mambairv2
```

### 1.3 Install PyTorch

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

If using a newer CUDA version (e.g. 12.x), match the PyTorch CUDA version accordingly and see the mamba_ssm and causal_conv_1d repos for compatible wheel versions.

### 1.4 Install Mamba Dependencies

Three possible approaches (try in order):

**Option A — pip install (preferred)**:
```bash
pip install causal_conv1d==1.1.1
pip install mamba_ssm==1.1.1
```

**Option B — from source**:
```bash
pip install causal-conv1d>=1.1.0
pip install mamba-ssm --no-build-isolation
```

**Option C — alternative selective scan**:
If mamba_ssm fails to install, the repo includes a pure-PyTorch fallback. Check `basicsr/archs/mambairv2_arch.py` — it may use `selective_scan_fn` from mamba_ssm or a local implementation.

### 1.5 Install BasicSR and Other Dependencies

```bash
pip install basicsr==1.3.5
pip install -r requirements.txt  # if present
pip install einops timm
```

### 1.6 Install the Project

```bash
pip install -e .
```

---

## Phase 2: Data Preparation

### 2.1 Symlink or Copy Data

The MambaIR codebase uses BasicSR's `PairedImageDataset`. Place data so the config can find it:

```bash
# From inside MambaIR repo root
mkdir -p datasets/TPAF_denoise

# Symlink your data
ln -s /absolute/path/to/train_data/train/clean datasets/TPAF_denoise/train_clean
ln -s /absolute/path/to/train_data/train/noisy datasets/TPAF_denoise/train_noisy
ln -s /absolute/path/to/train_data/val/clean datasets/TPAF_denoise/val_clean
ln -s /absolute/path/to/train_data/val/noisy datasets/TPAF_denoise/val_noisy
```

### 2.2 Verify Data Alignment

```python
import os
train_clean = sorted(os.listdir('datasets/TPAF_denoise/train_clean'))
train_noisy = sorted(os.listdir('datasets/TPAF_denoise/train_noisy'))
assert len(train_clean) == len(train_noisy) == 4488, f"Train count mismatch: {len(train_clean)} clean, {len(train_noisy)} noisy"
assert train_clean == train_noisy, "Filename mismatch between clean and noisy"

val_clean = sorted(os.listdir('datasets/TPAF_denoise/val_clean'))
val_noisy = sorted(os.listdir('datasets/TPAF_denoise/val_noisy'))
assert len(val_clean) == len(val_noisy) == 792, f"Val count mismatch: {len(val_clean)} clean, {len(val_noisy)} noisy"
assert val_clean == val_noisy, "Filename mismatch between clean and noisy val"
print("All checks passed.")
```

### 2.3 Grayscale Handling — IMPORTANT

MambaIRv2 defaults to `in_chans=3` (RGB). Our data is **single-channel grayscale**. Two approaches:

**Approach A (recommended): Set `in_chans=1` in the config.**
This is cleaner and more memory-efficient. The model natively supports `in_chans=1`.

**Approach B: Convert grayscale images to 3-channel by duplicating.**
Use a preprocessing script to save each grayscale image as a 3-channel PNG (R=G=B=gray value). Then keep `in_chans=3`. This is wasteful but avoids any code changes.

**Use Approach A** unless you encounter issues.

---

## Phase 3: Training Configuration

### 3.1 Create the Training Config YAML

Create file: `options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml`

```yaml
# --------------------------------------------------------------------------
# MambaIRv2 Denoising — TPAF Microscopy (Grayscale 512x512)
# --------------------------------------------------------------------------
name: MambaIRv2_TPAF_Denoise
model_type: MambaIRv2Model
scale: 1        # denoising is scale=1 (no upsampling)
num_gpu: 1      # adjust to your GPU count
manual_seed: 42

# ---------- Dataset ----------
datasets:
  train:
    task: DN       # denoising task
    name: TPAF_train
    type: PairedImageDataset

    dataroot_gt: ./datasets/TPAF_denoise/train_clean
    dataroot_lq: ./datasets/TPAF_denoise/train_noisy
    filename_tmpl: '{}'

    io_backend:
      type: disk

    gt_size: 128         # crop size during training (128x128 patches from 512x512)
    use_hflip: true
    use_rot: true

    # Dataloader
    use_shuffle: true
    num_worker_per_gpu: 8
    batch_size_per_gpu: 4   # adjust based on GPU VRAM (4 for 24GB, 2 for 12GB)
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

  val:
    task: DN
    name: TPAF_val
    type: PairedImageDataset

    dataroot_gt: ./datasets/TPAF_denoise/val_clean
    dataroot_lq: ./datasets/TPAF_denoise/val_noisy
    filename_tmpl: '{}'

    io_backend:
      type: disk

# ---------- Network Architecture ----------
network_g:
  type: MambaIRv2
  upscale: 1              # no upsampling for denoising
  in_chans: 1             # single-channel grayscale input
  img_size: 128           # must match gt_size
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
  upsampler: ''           # empty string = no upsampler (denoising)
  resi_connection: '1conv'

# ---------- Paths ----------
path:
  pretrain_network_g: ~             # set to a .pth path to finetune from pretrained
  strict_load_g: true
  resume_state: ~                   # set to a .state path to resume interrupted training

# ---------- Training Settings ----------
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

  total_iter: 400000
  warmup_iter: -1         # no warmup; set to a positive int to enable

  # Losses — Charbonnier loss for denoising (as per MambaIRv2 paper)
  pixel_opt:
    type: CharbonnierLoss
    loss_weight: 1.0
    reduction: mean

# ---------- Validation Settings ----------
val:
  val_freq: !!float 5e3   # validate every 5000 iterations
  save_img: false          # set true to save denoised validation images

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
      test_y_channel: false    # false for grayscale (no Y-channel extraction needed)
    ssim:
      type: calculate_ssim
      crop_border: 0
      test_y_channel: false

# ---------- Logging ----------
logger:
  print_freq: 200
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# ---------- Dist Training ----------
dist_params:
  backend: nccl
```

### 3.2 Key Config Considerations

- **`in_chans: 1`** — Critical for grayscale. If this causes errors in the architecture code, fall back to Approach B (3-channel duplicate) and set `in_chans: 3`.
- **`upscale: 1`** and **`upsampler: ''`** — Essential for denoising (no spatial upscaling).
- **`gt_size: 128`** — Random 128×128 crops from 512×512 during training. The paper uses 128×128 for denoising. You can increase to 256 if VRAM allows, but adjust `img_size` to match.
- **`img_size`** must match **`gt_size`**.
- **`batch_size_per_gpu`** — Start with 4 on a 24GB GPU. Reduce to 2 if OOM. For multi-GPU, total batch = batch_size_per_gpu × num_gpu.
- **`depths` and `embed_dim`** — The default denoising config uses `embed_dim=180` with 6 ASSB groups of depth 6. This matches the "Base" model. For a lighter model, try `embed_dim=48`, `depths=[6,6,6,6]`.
- **`total_iter: 400000`** — Standard for denoising. With 4488 images, batch_size=4, this is ~356 epochs. Adjust if needed.
- **`test_y_channel: false`** — Since our images are grayscale, we compute PSNR/SSIM on the full image, not just the Y channel.

### 3.3 Model Type Registration Check

Verify that `MambaIRv2Model` exists in the codebase. Check:
```bash
grep -r "MambaIRv2Model" basicsr/
```

If the model type is named differently (e.g., `ImageRestorationModel` or `MambaIRModel`), update the `model_type` field accordingly. The standard BasicSR denoising model is often `ImageCleanModel`. Look at existing denoising config files for the correct name:
```bash
ls options/train/mambairv2/
cat options/train/mambairv2/train_MambaIRv2_ColorDN_15.yml  # reference config
```

**Use whatever `model_type` the existing denoising config uses.** This is critical — copy it exactly.

---

## Phase 4: Training

### 4.1 Single-GPU Training

```bash
cd MambaIR
python basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml
```

### 4.2 Multi-GPU Training (Distributed)

```bash
python -m torch.distributed.launch \
  --nproc_per_node=NUM_GPUS \
  --master_port=4321 \
  basicsr/train.py \
  -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml \
  --launcher pytorch
```

Replace `NUM_GPUS` with your GPU count (e.g. 2, 4, 8).

### 4.3 Resume Training

If training is interrupted:
```bash
python basicsr/train.py \
  -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml \
  --auto_resume
```

Or manually set `resume_state` in the YAML to the latest `.state` file in `experiments/MambaIRv2_TPAF_Denoise/training_states/`.

### 4.4 Monitor Training

- **TensorBoard**: `tensorboard --logdir experiments/MambaIRv2_TPAF_Denoise/tb_logger`
- **Log file**: `experiments/MambaIRv2_TPAF_Denoise/train_MambaIRv2_TPAF_Denoise_*.log`
- **Checkpoints saved to**: `experiments/MambaIRv2_TPAF_Denoise/models/`

### 4.5 Expected Training Behavior

- Loss (`l_pix`) should decrease steadily from ~0.05 to ~0.005–0.01 range.
- Validation PSNR should increase over time. Good denoising results are typically 30+ dB PSNR.
- If loss plateaus early or validation PSNR is very low (<25 dB), check data loading: images might not be normalized correctly.

---

## Phase 5: Testing / Inference Configuration

### 5.1 Create the Test Config YAML

Create file: `options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml`

```yaml
# --------------------------------------------------------------------------
# MambaIRv2 Denoising Test — TPAF Microscopy
# --------------------------------------------------------------------------
name: MambaIRv2_TPAF_Denoise_Test
model_type: MambaIRv2Model    # must match training config
scale: 1
num_gpu: 1

datasets:
  test_1:
    task: DN
    name: TPAF_val
    type: PairedImageDataset

    dataroot_gt: ./datasets/TPAF_denoise/val_clean
    dataroot_lq: ./datasets/TPAF_denoise/val_noisy
    filename_tmpl: '{}'

    io_backend:
      type: disk

network_g:
  type: MambaIRv2
  upscale: 1
  in_chans: 1
  img_size: 512           # FULL resolution at test time (not cropped)
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
  pretrain_network_g: experiments/MambaIRv2_TPAF_Denoise/models/net_g_latest.pth
  # Or specify the best checkpoint: net_g_400000.pth

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
```

### 5.2 Key Differences from Training Config

- **`img_size: 512`** — At test time, use full image resolution (512×512), not crops.
- **`save_img: true`** — Saves denoised output images.
- **No training settings** — Only network, dataset, and path.

### 5.3 Important: img_size at Test Time

MambaIRv2 uses `img_size` to compute `patches_resolution` for positional embeddings and window partitioning. If 512 causes issues (OOM or dimension errors), use tiled/sliding-window inference or reduce to a divisible size. The `window_size=16` means `img_size` must be divisible by 16. Since 512 / 16 = 32, this is fine.

If VRAM is insufficient for full 512×512 inference, use overlapping tile-based inference (see Phase 6.3).

---

## Phase 6: Running Inference

### 6.1 Standard Test with Metrics

```bash
python basicsr/test.py -opt options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml
```

Results (PSNR, SSIM) printed to console and saved to `results/MambaIRv2_TPAF_Denoise_Test/`.

### 6.2 Inference on New (Unpaired) Noisy Images

For inference on new noisy images without ground truth, create a standalone script:

```python
"""
inference_tpaf.py — Denoise TPAF microscopy images using trained MambaIRv2.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from basicsr.archs.mambairv2_arch import MambaIRv2

def load_model(checkpoint_path, device='cuda'):
    """Load trained MambaIRv2 model."""
    model = MambaIRv2(
        upscale=1,
        in_chans=1,
        img_size=512,        # match test config
        img_range=1.0,
        embed_dim=180,
        d_state=8,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=2,
        upsampler='',
        resi_connection='1conv',
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'params' in state_dict:
        state_dict = state_dict['params']

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def denoise_image(model, img_path, device='cuda'):
    """Denoise a single grayscale image."""
    img = Image.open(img_path).convert('L')  # ensure grayscale
    img_np = np.array(img).astype(np.float32) / 255.0

    # Shape: (1, 1, H, W)
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze().clamp(0, 1).cpu().numpy()
    output_uint8 = (output * 255.0).round().astype(np.uint8)
    return output_uint8


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory of noisy input images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save denoised images')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(args.checkpoint, args.device)
    print(f"Model loaded from {args.checkpoint}")

    img_paths = sorted(
        glob.glob(os.path.join(args.input_dir, '*'))
    )
    img_paths = [p for p in img_paths if p.lower().endswith(('.png', '.tif', '.tiff', '.jpg', '.bmp'))]

    print(f"Found {len(img_paths)} images to denoise.")

    for i, img_path in enumerate(img_paths):
        fname = os.path.basename(img_path)
        output = denoise_image(model, img_path, args.device)
        out_path = os.path.join(args.output_dir, fname)

        # Save as PNG (lossless)
        Image.fromarray(output, mode='L').save(out_path)

        if (i + 1) % 50 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1}/{len(img_paths)}] Saved: {out_path}")

    print("Inference complete.")

if __name__ == '__main__':
    main()
```

**Usage:**
```bash
python inference_tpaf.py \
  --checkpoint experiments/MambaIRv2_TPAF_Denoise/models/net_g_latest.pth \
  --input_dir /path/to/new_noisy_images/ \
  --output_dir /path/to/denoised_output/
```

### 6.3 Tiled Inference (if OOM on Full 512×512)

If full-image inference causes OOM, implement tiled inference with overlap:

```python
def denoise_tiled(model, img_tensor, tile_size=256, overlap=32, device='cuda'):
    """Tile-based inference with overlap blending."""
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

**Note:** When using tiled inference, the model's `img_size` parameter should match `tile_size`, not 512.

---

## Phase 7: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `in_chans=1` causes shape errors | Switch to Approach B: duplicate grayscale to 3 channels, set `in_chans=3` |
| `MambaIRv2Model` not found | Check existing denoising config for correct `model_type` name. Try `ImageCleanModel` or `MambaIRModel` |
| OOM during training | Reduce `batch_size_per_gpu` to 2 or 1. Reduce `gt_size` to 64 (and `img_size` to 64) |
| OOM during inference | Use tiled inference (Phase 6.3) with `tile_size=256` |
| `mamba_ssm` install fails | Try building from source. Check CUDA version compatibility. As last resort, try the pure-PyTorch scan in the repo |
| Validation PSNR not improving | Check that clean images are actually clean (not swapped). Check image normalization (should be 0–255 uint8 on disk) |
| `window_size` assertion error | Ensure `img_size` is divisible by `window_size` (16). 128/16=8 ✓, 512/16=32 ✓ |
| Images look unchanged after inference | Model might not have converged. Check that loss decreased. Ensure the correct checkpoint is loaded |

### Verifying Data Loading

Add this to check that BasicSR reads your images correctly:
```python
from basicsr.data.paired_image_dataset import PairedImageDataset
opt = {
    'dataroot_gt': './datasets/TPAF_denoise/train_clean',
    'dataroot_lq': './datasets/TPAF_denoise/train_noisy',
    'filename_tmpl': '{}',
    'io_backend': {'type': 'disk'},
    'gt_size': 128,
    'use_hflip': True,
    'use_rot': True,
    'phase': 'train',
    'scale': 1,
}
ds = PairedImageDataset(opt)
sample = ds[0]
print(f"GT shape: {sample['gt'].shape}, LQ shape: {sample['lq'].shape}")
# Expected: GT shape: torch.Size([1, 128, 128]) for grayscale
# If shape is [3, 128, 128], BasicSR auto-converted to RGB — set in_chans=3
```

**IMPORTANT**: BasicSR's `imread` may automatically convert grayscale to 3-channel. If `sample['gt'].shape[0] == 3`, then set `in_chans: 3` in the config regardless and optionally add `color: y` in the dataset config to handle this.

---

## Phase 8: Evaluation & Results

### 8.1 Quantitative Metrics

After testing, results are saved in:
```
results/MambaIRv2_TPAF_Denoise_Test/
├── metrics.csv       # per-image PSNR, SSIM
└── visualization/    # denoised images (if save_img: true)
```

### 8.2 Compute Additional Metrics

```python
"""compute_metrics.py — PSNR, SSIM, and LPIPS on denoised vs clean."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image
import os

def evaluate(clean_dir, denoised_dir):
    psnrs, ssims = [], []
    for fname in sorted(os.listdir(clean_dir)):
        clean = np.array(Image.open(os.path.join(clean_dir, fname)).convert('L'))
        denoised = np.array(Image.open(os.path.join(denoised_dir, fname)).convert('L'))
        psnrs.append(peak_signal_noise_ratio(clean, denoised, data_range=255))
        ssims.append(structural_similarity(clean, denoised, data_range=255))
    print(f"PSNR: {np.mean(psnrs):.4f} ± {np.std(psnrs):.4f}")
    print(f"SSIM: {np.mean(ssims):.4f} ± {np.std(ssims):.4f}")
```

---

## Quick Reference Commands

```bash
# Train (single GPU)
python basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml

# Train (multi-GPU)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 \
  basicsr/train.py -opt options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml --launcher pytorch

# Test with metrics
python basicsr/test.py -opt options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml

# Inference on new images
python inference_tpaf.py --checkpoint experiments/MambaIRv2_TPAF_Denoise/models/net_g_latest.pth \
  --input_dir /path/to/noisy/ --output_dir /path/to/denoised/

# TensorBoard monitoring
tensorboard --logdir experiments/MambaIRv2_TPAF_Denoise/tb_logger
```

---

## File Checklist

After setup, you should have created these files:

- [ ] `options/train/mambairv2/train_MambaIRv2_TPAF_denoise.yml`
- [ ] `options/test/mambairv2/test_MambaIRv2_TPAF_denoise.yml`
- [ ] `inference_tpaf.py`
- [ ] `datasets/TPAF_denoise/` (symlinks to your data)

---

## Notes on Adapting from Color Denoising to Grayscale

The original MambaIRv2 denoising configs are for Gaussian **color** image denoising (3-channel, synthetic noise at known sigma levels like σ=15, 25, 50). Our task differs in two key ways:

1. **Grayscale (1-channel)** — handled by `in_chans=1` or by duplicating channels.
2. **Real noise (paired data)** — we use real clean-noisy pairs, not synthetic Gaussian noise. The training pipeline with `PairedImageDataset` and Charbonnier loss handles this correctly. No noise injection is needed during training because the noisy images are already provided.

The loss function (Charbonnier) and optimizer settings (Adam, lr=2e-4, MultiStepLR) match the paper's recommendations for denoising and should work well for this task.
