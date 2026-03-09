"""
generate_notebooks.py
Generates train.ipynb and inference.ipynb for MambaIRv2 TPAF denoising on Google Colab.
Run with: python generate_notebooks.py
"""
import json, os

# ──────────────────────────────────────────────────────────────────────────────
# Helper: build a cell dict
# ──────────────────────────────────────────────────────────────────────────────
_cell_id = 0
def _id():
    global _cell_id
    _cell_id += 1
    return f"cell-{_cell_id:03d}"

def md(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass  # last line has no trailing newline — correct for ipynb
    return {
        "cell_type": "markdown",
        "id": _id(),
        "metadata": {},
        "source": lines,
    }

def code(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": _id(),
        "metadata": {},
        "outputs": [],
        "source": lines,
    }

def notebook(cells: list) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
                "gpuType": "T4",
            },
            "accelerator": "GPU",
        },
        "cells": cells,
    }

def save(nb: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN.IPYNB
# ══════════════════════════════════════════════════════════════════════════════
_cell_id = 0

train_cells = []

# ── Title ──────────────────────────────────────────────────────────────────
train_cells.append(md("""\
# MambaIRv2 TPAF Denoising — Training Notebook (Phases 1–4)

This notebook trains **MambaIRv2** (CVPR 2025) as a denoising model for \
two-photon autofluorescence (TPAF) microscopy images.

**Workflow:**
1. Mount Google Drive
2. Clone the MambaIR repository & install all dependencies
3. Copy training data from Drive to local Colab storage (faster I/O)
4. Auto-detect `in_chans` and `model_type` from the actual data & repo
5. Programmatically generate the training YAML config
6. Symlink the `experiments/` directory to Drive (checkpoint persistence)
7. Start training

A **separate resume section** at the bottom lets you restart after a Colab disconnect.\
"""))

# ── 0. User Configuration ───────────────────────────────────────────────────
train_cells.append(md("## 0  User Configuration\nEdit **only this cell** before running the notebook."))
train_cells.append(code("""\
# ============================================================
# USER CONFIGURATION — edit these values before running
# ============================================================

# --- Google Drive paths ---
DRIVE_DATA_ROOT  = "/content/drive/MyDrive/TPAF_data"
# ^ Must contain:  train/clean/  train/noisy/  val/clean/  val/noisy/

DRIVE_EXPR_ROOT  = "/content/drive/MyDrive/MambaIR_experiments"
# ^ Checkpoints are symlinked here so they survive Colab disconnects.

# --- Local (Colab) paths ---
LOCAL_DATA_ROOT  = "/content/data"    # fast local SSD for data
MAMBAIR_DIR      = "/content/MambaIR" # where the repo is cloned

# --- Training hyper-parameters ---
BATCH_SIZE  = 4        # per GPU; reduce to 2 if OOM
GT_SIZE     = 128      # training crop size (img_size in config must match)
TOTAL_ITER  = 400000   # total training iterations
NUM_GPU     = 1        # number of GPUs
MANUAL_SEED = 42

LR          = "2e-4"   # learning rate (kept as string for YAML !!float tag)

# --- Model architecture (leave as-is for MambaIRv2-Base) ---
EMBED_DIM = 180
DEPTHS    = [6, 6, 6, 6, 6, 6]
NUM_HEADS = [6, 6, 6, 6, 6, 6]

# ============================================================
print("Configuration loaded.")
print(f"  Drive data root : {DRIVE_DATA_ROOT}")
print(f"  Drive expr root : {DRIVE_EXPR_ROOT}")
print(f"  Local data root : {LOCAL_DATA_ROOT}")
print(f"  MambaIR dir     : {MAMBAIR_DIR}")
print(f"  Batch size      : {BATCH_SIZE}  |  GT crop : {GT_SIZE}  |  Total iter : {TOTAL_ITER}")\
"""))

# ── 1. Mount Google Drive ───────────────────────────────────────────────────
train_cells.append(md("## 1  Mount Google Drive"))
train_cells.append(code("""\
from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted at /content/drive")\
"""))

# ── 2. Clone MambaIR ────────────────────────────────────────────────────────
train_cells.append(md("""\
## 2  Clone the MambaIR Repository

The **main branch** of [csguoh/MambaIR](https://github.com/csguoh/MambaIR) contains MambaIRv2.\
"""))
train_cells.append(code("""\
import os, subprocess

if not os.path.exists(MAMBAIR_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/csguoh/MambaIR.git", MAMBAIR_DIR],
        check=True,
    )
    print(f"Cloned MambaIR to {MAMBAIR_DIR}")
else:
    print(f"Repo already exists at {MAMBAIR_DIR} — skipping clone.")

os.chdir(MAMBAIR_DIR)
print(f"Working directory: {os.getcwd()}")\
"""))

# ── 3. Install Dependencies ──────────────────────────────────────────────────
train_cells.append(md("""\
## 3  Install Dependencies (pip only)

> **Note:** `mamba_ssm` compilation takes ~3–5 minutes on Colab.
> If `mamba_ssm==1.1.1` fails to build, the cell tries the latest version from source.\
"""))
train_cells.append(code("""\
import subprocess, sys

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

# 1) PyTorch with CUDA 11.8 wheels (compatible with Colab's CUDA runtime)
pip("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
    "--index-url", "https://download.pytorch.org/whl/cu118")

# 2) Mamba SSM dependencies
try:
    pip("causal_conv1d==1.1.1")
    pip("mamba_ssm==1.1.1")
    print("mamba_ssm 1.1.1 installed successfully.")
except Exception:
    print("Fixed-version install failed — trying latest from source...")
    pip("causal-conv1d>=1.1.0")
    pip("mamba-ssm", "--no-build-isolation")

# 3) BasicSR + extras
pip("basicsr==1.3.5", "einops", "timm")

# 4) Install the MambaIR repo itself (editable)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

print("\\nAll dependencies installed.")\
"""))

# ── 4. Copy Data from Drive ──────────────────────────────────────────────────
train_cells.append(md("""\
## 4  Copy Data from Google Drive to Local Storage

Copying to `/content/` gives ~10× faster I/O compared with reading directly from Drive.\
"""))
train_cells.append(code("""\
import shutil, os

LOCAL_TRAIN_CLEAN = f"{LOCAL_DATA_ROOT}/train/clean"
LOCAL_TRAIN_NOISY = f"{LOCAL_DATA_ROOT}/train/noisy"
LOCAL_VAL_CLEAN   = f"{LOCAL_DATA_ROOT}/val/clean"
LOCAL_VAL_NOISY   = f"{LOCAL_DATA_ROOT}/val/noisy"

pairs = [
    (f"{DRIVE_DATA_ROOT}/train/clean", LOCAL_TRAIN_CLEAN),
    (f"{DRIVE_DATA_ROOT}/train/noisy", LOCAL_TRAIN_NOISY),
    (f"{DRIVE_DATA_ROOT}/val/clean",   LOCAL_VAL_CLEAN),
    (f"{DRIVE_DATA_ROOT}/val/noisy",   LOCAL_VAL_NOISY),
]

for src, dst in pairs:
    if os.path.exists(dst):
        print(f"Already exists — skipping: {dst}")
    else:
        print(f"Copying {src}  →  {dst} ...")
        shutil.copytree(src, dst)
        print(f"  Done.")

print("\\nData copy complete.")\
"""))

# ── 5. Verify Data Alignment ────────────────────────────────────────────────
train_cells.append(md("""\
## 5  Verify Data Alignment

Clean and noisy filenames **must be identical and in the same order**.\
"""))
train_cells.append(code("""\
train_clean = sorted(f for f in os.listdir(LOCAL_TRAIN_CLEAN) if not f.startswith('.'))
train_noisy = sorted(f for f in os.listdir(LOCAL_TRAIN_NOISY) if not f.startswith('.'))
val_clean   = sorted(f for f in os.listdir(LOCAL_VAL_CLEAN)   if not f.startswith('.'))
val_noisy   = sorted(f for f in os.listdir(LOCAL_VAL_NOISY)   if not f.startswith('.'))

assert len(train_clean) == len(train_noisy), (
    f"Train count mismatch: {len(train_clean)} clean vs {len(train_noisy)} noisy")
assert train_clean == train_noisy, (
    f"Train filename mismatch!\\n  First diff: {next((a,b) for a,b in zip(train_clean,train_noisy) if a!=b)}")
assert len(val_clean) == len(val_noisy), (
    f"Val count mismatch: {len(val_clean)} clean vs {len(val_noisy)} noisy")
assert val_clean == val_noisy, "Val filename mismatch!"

print(f"Train pairs : {len(train_clean)}")
print(f"Val pairs   : {len(val_clean)}")
print("Data alignment check PASSED.")\
"""))

# ── 6. Auto-detect in_chans ──────────────────────────────────────────────────
train_cells.append(md("""\
## 6  Auto-detect `in_chans`

BasicSR's `imread` may silently convert grayscale images to 3-channel RGB.
We load one sample through the actual data pipeline to discover the real channel count.\
"""))
train_cells.append(code("""\
import sys
sys.path.insert(0, MAMBAIR_DIR)
from basicsr.data.paired_image_dataset import PairedImageDataset

_opt = {
    "dataroot_gt"   : LOCAL_TRAIN_CLEAN,
    "dataroot_lq"   : LOCAL_TRAIN_NOISY,
    "filename_tmpl" : "{}",
    "io_backend"    : {"type": "disk"},
    "gt_size"       : 128,
    "use_hflip"     : False,
    "use_rot"       : False,
    "phase"         : "train",
    "scale"         : 1,
}
_ds     = PairedImageDataset(_opt)
_sample = _ds[0]
IN_CHANS = int(_sample["gt"].shape[0])   # 1 = grayscale kept as-is, 3 = auto-converted to RGB
del _ds, _sample, _opt

print(f"Detected in_chans = {IN_CHANS}")
if IN_CHANS == 1:
    print("  → Grayscale images loaded as single-channel (ideal).")
else:
    print("  → BasicSR converted grayscale to 3-channel RGB.")
    print("    The config will use in_chans=3 accordingly.")\
"""))

# ── 7. Auto-detect model_type ────────────────────────────────────────────────
train_cells.append(md("""\
## 7  Auto-detect `model_type`

The correct `model_type` string is read from the existing MambaIR denoising configs \
so our YAML is always consistent with the installed codebase.\
"""))
train_cells.append(code("""\
import glob, re

MODEL_TYPE = None
search_patterns = [
    f"{MAMBAIR_DIR}/options/train/mambairv2/*DN*.yml",
    f"{MAMBAIR_DIR}/options/train/mambairv2/*denois*.yml",
    f"{MAMBAIR_DIR}/options/train/mambairv2/*.yml",
]

for pattern in search_patterns:
    for yml_path in sorted(glob.glob(pattern)):
        with open(yml_path) as fh:
            text = fh.read()
        m = re.search(r'^model_type:\\s*(\\S+)', text, re.MULTILINE)
        if m:
            MODEL_TYPE = m.group(1)
            print(f"Found model_type '{MODEL_TYPE}'")
            print(f"  Source: {yml_path}")
            break
    if MODEL_TYPE:
        break

if MODEL_TYPE is None:
    MODEL_TYPE = "ImageCleanModel"
    print(f"model_type not found in existing configs — defaulting to '{MODEL_TYPE}'")

print(f"\\nUsing model_type = {MODEL_TYPE}")\
"""))

# ── 8. Generate Training YAML ────────────────────────────────────────────────
train_cells.append(md("""\
## 8  Generate Training YAML Config

The config is written programmatically using the auto-detected `IN_CHANS` and `MODEL_TYPE` values.\
"""))
train_cells.append(code('''\
import os

yml_dir        = f"{MAMBAIR_DIR}/options/train/mambairv2"
TRAIN_YML_PATH = f"{yml_dir}/train_MambaIRv2_TPAF_denoise.yml"
os.makedirs(yml_dir, exist_ok=True)

TRAIN_YML = f"""# -----------------------------------------------------------------------
# MambaIRv2 Denoising — TPAF Microscopy (auto-generated by train.ipynb)
# -----------------------------------------------------------------------
name: MambaIRv2_TPAF_Denoise
model_type: {MODEL_TYPE}
scale: 1
num_gpu: {NUM_GPU}
manual_seed: {MANUAL_SEED}

# ---------- Datasets ----------
datasets:
  train:
    task: DN
    name: TPAF_train
    type: PairedImageDataset

    dataroot_gt: {LOCAL_TRAIN_CLEAN}
    dataroot_lq: {LOCAL_TRAIN_NOISY}
    filename_tmpl: \'{{}}\'

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

    dataroot_gt: {LOCAL_VAL_CLEAN}
    dataroot_lq: {LOCAL_VAL_NOISY}
    filename_tmpl: \'{{}}\'

    io_backend:
      type: disk

# ---------- Network ----------
network_g:
  type: MambaIRv2
  upscale: 1
  in_chans: {IN_CHANS}
  img_size: {GT_SIZE}
  img_range: 1.0
  embed_dim: {EMBED_DIM}
  d_state: 8
  depths: {DEPTHS}
  num_heads: {NUM_HEADS}
  window_size: 16
  inner_rank: 32
  num_tokens: 64
  convffn_kernel_size: 5
  mlp_ratio: 2
  upsampler: \'\'
  resi_connection: \'1conv\'

# ---------- Paths ----------
path:
  pretrain_network_g: ~
  strict_load_g: true
  resume_state: ~

# ---------- Training ----------
train:
  ema_decay: 0.999

  optim_g:
    type: Adam
    lr: !!float {LR}
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

# ---------- Validation ----------
val:
  val_freq: !!float 5e3
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

# ---------- Logging ----------
logger:
  print_freq: 200
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# ---------- Distributed ----------
dist_params:
  backend: nccl
"""

with open(TRAIN_YML_PATH, "w") as fh:
    fh.write(TRAIN_YML)

print(f"Training config written to:")
print(f"  {TRAIN_YML_PATH}")
print(f"\\nKey settings:")
print(f"  model_type = {MODEL_TYPE}")
print(f"  in_chans   = {IN_CHANS}")
print(f"  gt_size    = {GT_SIZE}  (img_size matches)")
print(f"  batch_size = {BATCH_SIZE}")
print(f"  total_iter = {TOTAL_ITER}")
'''))

# ── 9. Symlink experiments/ to Drive ─────────────────────────────────────────
train_cells.append(md("""\
## 9  Symlink `experiments/` to Google Drive

Colab VMs reset on disconnect. By symlinking the `experiments/` directory to your \
Drive, **checkpoints are written directly to Drive** and survive disconnects.\
"""))
train_cells.append(code("""\
import os, shutil

LOCAL_EXPR_PATH = f"{MAMBAIR_DIR}/experiments"

# Ensure the Drive target directory exists
os.makedirs(DRIVE_EXPR_ROOT, exist_ok=True)

if os.path.islink(LOCAL_EXPR_PATH):
    current_target = os.readlink(LOCAL_EXPR_PATH)
    print(f"Symlink already exists: {LOCAL_EXPR_PATH} → {current_target}")

elif os.path.isdir(LOCAL_EXPR_PATH):
    print(f"{LOCAL_EXPR_PATH} exists as a real directory — migrating to Drive ...")
    shutil.copytree(LOCAL_EXPR_PATH, DRIVE_EXPR_ROOT, dirs_exist_ok=True)
    shutil.rmtree(LOCAL_EXPR_PATH)
    os.symlink(DRIVE_EXPR_ROOT, LOCAL_EXPR_PATH)
    print(f"Migrated and symlinked: {LOCAL_EXPR_PATH} → {DRIVE_EXPR_ROOT}")

else:
    os.symlink(DRIVE_EXPR_ROOT, LOCAL_EXPR_PATH)
    print(f"Symlinked: {LOCAL_EXPR_PATH} → {DRIVE_EXPR_ROOT}")

print("\\nCheckpoints will be saved to Drive automatically.")\
"""))

# ── 10. TensorBoard ──────────────────────────────────────────────────────────
train_cells.append(md("""\
## 10  TensorBoard Monitoring (optional)

Run this cell **before or after** starting training to view live loss and PSNR curves.
The TensorBoard widget updates automatically while training runs in the cell below.\
"""))
train_cells.append(code("""\
import os
TB_LOG_DIR = f"{MAMBAIR_DIR}/experiments/MambaIRv2_TPAF_Denoise/tb_logger"
os.makedirs(TB_LOG_DIR, exist_ok=True)

%load_ext tensorboard
%tensorboard --logdir {TB_LOG_DIR}\
"""))

# ── 11. Start Training ───────────────────────────────────────────────────────
train_cells.append(md("""\
## 11  Start Training

This cell runs training from scratch.
**If you need to resume after a disconnect, skip to the [Resume Section](#resume) below.**

> Expected behaviour:
> - `l_pix` loss should decrease from ~0.05 toward ~0.005–0.01
> - Validation PSNR should increase; good denoising results are ≥30 dB
> - Checkpoints saved every 5 000 iterations to Drive\
"""))
train_cells.append(code("""\
import os
os.chdir(MAMBAIR_DIR)

# Sanity-check that the config exists
assert os.path.exists(TRAIN_YML_PATH), f"Config not found: {TRAIN_YML_PATH}"
print(f"Starting training with config: {TRAIN_YML_PATH}\\n")

!python basicsr/train.py -opt {TRAIN_YML_PATH}\
"""))

# ── 12. Resume Section ───────────────────────────────────────────────────────
train_cells.append(md("""\
---
<a id="resume"></a>
## 12  Resume Training After a Colab Disconnect

After a Colab session resets:

1. **Re-run cells 0–9** (Configuration → Symlink `experiments/`) to restore the environment.
   Data copy (cell 4) will be skipped automatically if the local copies already exist.
2. **Run the cell below** — `--auto_resume` picks up the latest checkpoint from the
   `experiments/MambaIRv2_TPAF_Denoise/training_states/` directory automatically.

> **Tip:** You can also set `resume_state` manually in the generated YAML to a specific
> `.state` file if you want to resume from a particular checkpoint.\
"""))
train_cells.append(code("""\
import os
os.chdir(MAMBAIR_DIR)

# Verify checkpoint directory on Drive
import glob
state_files = sorted(glob.glob(
    f"{DRIVE_EXPR_ROOT}/MambaIRv2_TPAF_Denoise/training_states/*.state"
))
if state_files:
    print(f"Found {len(state_files)} training state(s). Latest:")
    print(f"  {state_files[-1]}")
else:
    print("No training states found — will start from scratch if --auto_resume finds nothing.")

print(f"\\nResuming training ...\\n")
!python basicsr/train.py -opt {TRAIN_YML_PATH} --auto_resume\
"""))

# ── Save train.ipynb ────────────────────────────────────────────────────────
save(notebook(train_cells), "train.ipynb")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE.IPYNB
# ══════════════════════════════════════════════════════════════════════════════
_cell_id = 0

inf_cells = []

# ── Title ──────────────────────────────────────────────────────────────────
inf_cells.append(md("""\
# MambaIRv2 TPAF Denoising — Inference & Evaluation Notebook (Phases 5–6, 8)

This notebook covers:

- **Paired evaluation** on validation data with PSNR & SSIM metrics (Phase 5–6)
- **Standalone inference** on new unpaired noisy images from Drive (Phase 6.2)
- **Tiled inference** fallback for OOM situations (Phase 6.3)
- **Visual side-by-side comparison**: noisy / denoised / ground truth (matplotlib)
- **Aggregate metrics** computation and results saved back to Drive (Phase 8)\
"""))

# ── 0. User Configuration ───────────────────────────────────────────────────
inf_cells.append(md("## 0  User Configuration\nEdit **only this cell** before running the notebook."))
inf_cells.append(code("""\
# ============================================================
# USER CONFIGURATION — edit these values before running
# ============================================================

# --- Google Drive paths ---
DRIVE_DATA_ROOT   = "/content/drive/MyDrive/TPAF_data"
# ^ Must contain:  val/clean/  val/noisy/

DRIVE_EXPR_ROOT   = "/content/drive/MyDrive/MambaIR_experiments"
# ^ Where training saved checkpoints (same as train.ipynb setting)

DRIVE_NOISY_DIR   = "/content/drive/MyDrive/TPAF_new_noisy"
# ^ New unpaired noisy images for standalone inference (can be same as val/noisy)

DRIVE_RESULTS_DIR = "/content/drive/MyDrive/TPAF_results"
# ^ Where denoised outputs and metrics will be saved

# --- Checkpoint ---
CHECKPOINT_PATH   = None
# Set to None to auto-select net_g_latest.pth from DRIVE_EXPR_ROOT.
# Or specify a path: "/content/drive/MyDrive/MambaIR_experiments/MambaIRv2_TPAF_Denoise/models/net_g_400000.pth"

# --- Local (Colab) paths ---
LOCAL_DATA_ROOT   = "/content/data"
LOCAL_NOISY_DIR   = "/content/new_noisy"   # copied from DRIVE_NOISY_DIR
LOCAL_RESULTS_DIR = "/content/results"
MAMBAIR_DIR       = "/content/MambaIR"

# --- Tiled inference (OOM fallback) ---
TILE_SIZE    = 256   # tile side length in pixels (must be divisible by window_size=16)
TILE_OVERLAP = 32    # overlap between adjacent tiles (pixels)

# --- Visualisation ---
VIS_NUM_IMAGES = 5   # how many val images to show in the side-by-side comparison

# --- Model architecture (must match what was used during training) ---
EMBED_DIM = 180
DEPTHS    = [6, 6, 6, 6, 6, 6]
NUM_HEADS = [6, 6, 6, 6, 6, 6]

# ============================================================
print("Configuration loaded.")
print(f"  Drive data root   : {DRIVE_DATA_ROOT}")
print(f"  Drive expr root   : {DRIVE_EXPR_ROOT}")
print(f"  Drive noisy dir   : {DRIVE_NOISY_DIR}")
print(f"  Drive results dir : {DRIVE_RESULTS_DIR}")
print(f"  Checkpoint        : {CHECKPOINT_PATH or 'auto-detect'}")\
"""))

# ── 1. Mount Google Drive ───────────────────────────────────────────────────
inf_cells.append(md("## 1  Mount Google Drive"))
inf_cells.append(code("""\
from google.colab import drive
drive.mount('/content/drive')
print("Drive mounted at /content/drive")\
"""))

# ── 2. Clone + Install ──────────────────────────────────────────────────────
inf_cells.append(md("""\
## 2  Clone MambaIR Repository & Install Dependencies

> `mamba_ssm` compilation takes ~3–5 minutes on first run.\
"""))
inf_cells.append(code("""\
import os, subprocess, sys

# Clone repo
if not os.path.exists(MAMBAIR_DIR):
    subprocess.run(
        ["git", "clone", "https://github.com/csguoh/MambaIR.git", MAMBAIR_DIR],
        check=True,
    )
    print(f"Cloned MambaIR to {MAMBAIR_DIR}")
else:
    print(f"Repo already exists at {MAMBAIR_DIR} — skipping clone.")

os.chdir(MAMBAIR_DIR)

def pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

pip("torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2",
    "--index-url", "https://download.pytorch.org/whl/cu118")

try:
    pip("causal_conv1d==1.1.1")
    pip("mamba_ssm==1.1.1")
    print("mamba_ssm 1.1.1 installed.")
except Exception:
    print("Fixed-version install failed — trying latest from source...")
    pip("causal-conv1d>=1.1.0")
    pip("mamba-ssm", "--no-build-isolation")

pip("basicsr==1.3.5", "einops", "timm")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)
pip("scikit-image")   # for PSNR / SSIM in Phase 8

print("\\nAll dependencies installed.")\
"""))

# ── 3. Copy Data from Drive ──────────────────────────────────────────────────
inf_cells.append(md("""\
## 3  Copy Data & New Noisy Images from Drive to Local Storage\
"""))
inf_cells.append(code("""\
import shutil, os

LOCAL_VAL_CLEAN = f"{LOCAL_DATA_ROOT}/val/clean"
LOCAL_VAL_NOISY = f"{LOCAL_DATA_ROOT}/val/noisy"

# Validation data (for paired evaluation)
for src, dst in [
    (f"{DRIVE_DATA_ROOT}/val/clean", LOCAL_VAL_CLEAN),
    (f"{DRIVE_DATA_ROOT}/val/noisy", LOCAL_VAL_NOISY),
]:
    if os.path.exists(dst):
        print(f"Already exists — skipping: {dst}")
    else:
        print(f"Copying {src}  →  {dst} ...")
        shutil.copytree(src, dst)

# New unpaired noisy images
if os.path.exists(DRIVE_NOISY_DIR):
    if os.path.exists(LOCAL_NOISY_DIR):
        print(f"Already exists — skipping: {LOCAL_NOISY_DIR}")
    else:
        print(f"Copying {DRIVE_NOISY_DIR}  →  {LOCAL_NOISY_DIR} ...")
        shutil.copytree(DRIVE_NOISY_DIR, LOCAL_NOISY_DIR)
else:
    print(f"DRIVE_NOISY_DIR not found ({DRIVE_NOISY_DIR}) — standalone inference will be skipped.")
    LOCAL_NOISY_DIR = None

os.makedirs(LOCAL_RESULTS_DIR, exist_ok=True)
print("\\nData ready.")\
"""))

# ── 4. Auto-detect in_chans ──────────────────────────────────────────────────
inf_cells.append(md("""\
## 4  Auto-detect `in_chans`

Loads one validation sample through BasicSR's pipeline to discover the real channel count.\
"""))
inf_cells.append(code("""\
import sys
sys.path.insert(0, MAMBAIR_DIR)
from basicsr.data.paired_image_dataset import PairedImageDataset

_opt = {
    "dataroot_gt"   : LOCAL_VAL_CLEAN,
    "dataroot_lq"   : LOCAL_VAL_NOISY,
    "filename_tmpl" : "{}",
    "io_backend"    : {"type": "disk"},
    "gt_size"       : 128,
    "use_hflip"     : False,
    "use_rot"       : False,
    "phase"         : "train",
    "scale"         : 1,
}
_ds     = PairedImageDataset(_opt)
_sample = _ds[0]
IN_CHANS = int(_sample["gt"].shape[0])
del _ds, _sample, _opt

print(f"Detected in_chans = {IN_CHANS}")
if IN_CHANS == 1:
    print("  → Grayscale kept as single-channel.")
else:
    print("  → BasicSR converted grayscale to 3-channel RGB.")\
"""))

# ── 5. Auto-detect model_type ────────────────────────────────────────────────
inf_cells.append(md("## 5  Auto-detect `model_type`"))
inf_cells.append(code("""\
import glob, re

MODEL_TYPE = None
for pattern in [
    f"{MAMBAIR_DIR}/options/train/mambairv2/*DN*.yml",
    f"{MAMBAIR_DIR}/options/train/mambairv2/*denois*.yml",
    f"{MAMBAIR_DIR}/options/train/mambairv2/*.yml",
]:
    for yml_path in sorted(glob.glob(pattern)):
        with open(yml_path) as fh:
            text = fh.read()
        m = re.search(r'^model_type:\\s*(\\S+)', text, re.MULTILINE)
        if m:
            MODEL_TYPE = m.group(1)
            print(f"Found model_type '{MODEL_TYPE}'  (from {yml_path})")
            break
    if MODEL_TYPE:
        break

if MODEL_TYPE is None:
    MODEL_TYPE = "ImageCleanModel"
    print(f"Defaulting to model_type = '{MODEL_TYPE}'")

print(f"\\nUsing model_type = {MODEL_TYPE}")\
"""))

# ── 6. Resolve Checkpoint ────────────────────────────────────────────────────
inf_cells.append(md("""\
## 6  Resolve Checkpoint Path

If `CHECKPOINT_PATH` is `None`, the notebook auto-selects `net_g_latest.pth` \
(or the highest-numbered checkpoint) from your Drive experiment directory.\
"""))
inf_cells.append(code("""\
import glob, os

EXPR_MODELS_DIR = f"{DRIVE_EXPR_ROOT}/MambaIRv2_TPAF_Denoise/models"

if CHECKPOINT_PATH is None:
    latest_path = f"{EXPR_MODELS_DIR}/net_g_latest.pth"
    if os.path.exists(latest_path):
        CHECKPOINT_PATH = latest_path
    else:
        candidates = sorted(glob.glob(f"{EXPR_MODELS_DIR}/net_g_*.pth"))
        assert candidates, (
            f"No checkpoints found in {EXPR_MODELS_DIR}.\\n"
            "Train the model first (train.ipynb) or set CHECKPOINT_PATH manually.")
        CHECKPOINT_PATH = candidates[-1]
        print(f"net_g_latest.pth not found — using: {os.path.basename(CHECKPOINT_PATH)}")

assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint not found: {CHECKPOINT_PATH}"
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"  Size: {os.path.getsize(CHECKPOINT_PATH) / 1e6:.1f} MB")\
"""))

# ── 7. Generate Test YAML ────────────────────────────────────────────────────
inf_cells.append(md("""\
## 7  Generate Test YAML Config

At test time, `img_size=512` (full image resolution).
`window_size=16` divides 512 evenly (512/16 = 32), so no padding is needed.\
"""))
inf_cells.append(code('''\
import os

yml_dir       = f"{MAMBAIR_DIR}/options/test/mambairv2"
TEST_YML_PATH = f"{yml_dir}/test_MambaIRv2_TPAF_denoise.yml"
os.makedirs(yml_dir, exist_ok=True)

TEST_RESULTS_DIR = f"{LOCAL_RESULTS_DIR}/paired_eval"
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

TEST_YML = f"""# -----------------------------------------------------------------------
# MambaIRv2 Denoising Test — TPAF Microscopy (auto-generated)
# -----------------------------------------------------------------------
name: MambaIRv2_TPAF_Denoise_Test
model_type: {MODEL_TYPE}
scale: 1
num_gpu: 1
manual_seed: 42

datasets:
  test_1:
    task: DN
    name: TPAF_val
    type: PairedImageDataset

    dataroot_gt: {LOCAL_VAL_CLEAN}
    dataroot_lq: {LOCAL_VAL_NOISY}
    filename_tmpl: \'{{}}\'

    io_backend:
      type: disk

network_g:
  type: MambaIRv2
  upscale: 1
  in_chans: {IN_CHANS}
  img_size: 512
  img_range: 1.0
  embed_dim: {EMBED_DIM}
  d_state: 8
  depths: {DEPTHS}
  num_heads: {NUM_HEADS}
  window_size: 16
  inner_rank: 32
  num_tokens: 64
  convffn_kernel_size: 5
  mlp_ratio: 2
  upsampler: \'\'
  resi_connection: \'1conv\'

path:
  pretrain_network_g: {CHECKPOINT_PATH}

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

with open(TEST_YML_PATH, "w") as fh:
    fh.write(TEST_YML)

print(f"Test config written to:")
print(f"  {TEST_YML_PATH}")
print(f"\\nKey settings:")
print(f"  model_type = {MODEL_TYPE}")
print(f"  in_chans   = {IN_CHANS}")
print(f"  img_size   = 512  (full resolution)")
print(f"  checkpoint = {CHECKPOINT_PATH}")
'''))

# ── 8. Run basicsr/test.py ───────────────────────────────────────────────────
inf_cells.append(md("""\
## 8  Run Paired Evaluation with `basicsr/test.py`

Evaluates the model on the validation set and reports per-image and average PSNR/SSIM.
Denoised images are saved under `results/MambaIRv2_TPAF_Denoise_Test/`.\
"""))
inf_cells.append(code("""\
import os
os.chdir(MAMBAIR_DIR)

print(f"Running evaluation with: {TEST_YML_PATH}\\n")
!python basicsr/test.py -opt {TEST_YML_PATH}\
"""))

# ── 9. Inference Functions ───────────────────────────────────────────────────
inf_cells.append(md("""\
## 9  Inference Helper Functions

Three functions:
- `load_model()` — loads the MambaIRv2 checkpoint
- `denoise_image()` — full-image inference
- `denoise_tiled()` — tiled inference (OOM fallback; overlapping tiles are blended)\
"""))
inf_cells.append(code("""\
import os, sys, glob
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, MAMBAIR_DIR)
from basicsr.archs.mambairv2_arch import MambaIRv2


def load_model(checkpoint_path: str, in_chans: int, device: str = "cuda") -> MambaIRv2:
    \"\"\"Load a trained MambaIRv2 checkpoint.\"\"\"
    model = MambaIRv2(
        upscale          = 1,
        in_chans         = in_chans,
        img_size         = 512,     # full-resolution at inference
        img_range        = 1.0,
        embed_dim        = EMBED_DIM,
        d_state          = 8,
        depths           = DEPTHS,
        num_heads        = NUM_HEADS,
        window_size      = 16,
        inner_rank       = 32,
        num_tokens       = 64,
        convffn_kernel_size = 5,
        mlp_ratio        = 2,
        upsampler        = '',
        resi_connection  = '1conv',
    )
    sd = torch.load(checkpoint_path, map_location=device)
    # Handle different checkpoint formats
    if isinstance(sd, dict):
        sd = sd.get("params_ema", sd.get("params", sd))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def _img_to_tensor(img_path: str, in_chans: int, device: str) -> torch.Tensor:
    \"\"\"Load a grayscale image as a float32 tensor of shape (1, C, H, W).\"\"\"
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W)
    if in_chans == 3:
        arr = np.stack([arr, arr, arr], axis=0)     # (3, H, W)
    else:
        arr = arr[np.newaxis]                        # (1, H, W)
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)


def _tensor_to_uint8(tensor: torch.Tensor, in_chans: int) -> np.ndarray:
    \"\"\"Convert model output tensor to uint8 numpy array (grayscale).\"\"\"
    out = tensor.squeeze()           # (C, H, W) or (H, W)
    if in_chans == 3 and out.ndim == 3:
        out = out.mean(dim=0)        # average 3 channels back to grayscale
    return (out.clamp(0.0, 1.0).cpu().numpy() * 255.0).round().astype(np.uint8)


def denoise_image(model: MambaIRv2, img_path: str,
                  in_chans: int, device: str = "cuda") -> np.ndarray:
    \"\"\"Full-image inference. May raise torch.cuda.OutOfMemoryError on large images.\"\"\"
    tensor = _img_to_tensor(img_path, in_chans, device)
    with torch.no_grad():
        output = model(tensor)
    return _tensor_to_uint8(output, in_chans)


def denoise_tiled(model: MambaIRv2, img_path: str, in_chans: int,
                  tile_size: int = TILE_SIZE, overlap: int = TILE_OVERLAP,
                  device: str = "cuda") -> np.ndarray:
    \"\"\"
    Tiled inference with overlap blending (OOM fallback).
    Overlapping regions are averaged, which smooths tile-boundary artefacts.
    tile_size must be divisible by window_size (16).
    \"\"\"
    assert tile_size % 16 == 0, f"tile_size must be divisible by 16, got {tile_size}"
    img_tensor = _img_to_tensor(img_path, in_chans, device)
    _, _, H, W = img_tensor.shape

    output = torch.zeros_like(img_tensor)
    weight = torch.zeros_like(img_tensor)
    stride = tile_size - overlap

    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y_end   = min(y + tile_size, H)
            x_end   = min(x + tile_size, W)
            y_start = y_end - tile_size    # ensure tile is always tile_size×tile_size
            x_start = x_end - tile_size

            tile = img_tensor[:, :, y_start:y_end, x_start:x_end]
            with torch.no_grad():
                tile_out = model(tile)

            output[:, :, y_start:y_end, x_start:x_end] += tile_out
            weight[:, :, y_start:y_end, x_start:x_end] += 1.0

    return _tensor_to_uint8(output / weight, in_chans)


print("Inference helper functions defined.")\
"""))

# ── 10. Load Model ────────────────────────────────────────────────────────────
inf_cells.append(md("## 10  Load Model"))
inf_cells.append(code("""\
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

model = load_model(CHECKPOINT_PATH, IN_CHANS, DEVICE)
print(f"Model loaded successfully.")
print(f"  in_chans   = {IN_CHANS}")
print(f"  checkpoint = {CHECKPOINT_PATH}")\
"""))

# ── 11. Inference on Unpaired Images ─────────────────────────────────────────
inf_cells.append(md("""\
## 11  Standalone Inference on Unpaired Noisy Images

Processes all images in `LOCAL_NOISY_DIR` (copied from `DRIVE_NOISY_DIR`).
Falls back to **tiled inference** automatically if a full-image forward pass runs out of GPU memory.\
"""))
inf_cells.append(code("""\
import glob, os
from pathlib import Path

STANDALONE_OUT = f"{LOCAL_RESULTS_DIR}/standalone"
os.makedirs(STANDALONE_OUT, exist_ok=True)

if LOCAL_NOISY_DIR is None or not os.path.exists(LOCAL_NOISY_DIR):
    print("LOCAL_NOISY_DIR not set or missing — skipping standalone inference.")
    print("Set DRIVE_NOISY_DIR in the configuration cell and re-run cell 3.")
else:
    EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp"}
    img_paths = sorted(
        p for p in glob.glob(f"{LOCAL_NOISY_DIR}/**/*", recursive=True)
        if Path(p).suffix.lower() in EXTS
    )
    print(f"Found {len(img_paths)} image(s) in {LOCAL_NOISY_DIR}")

    for i, img_path in enumerate(img_paths):
        fname = os.path.basename(img_path)
        try:
            out = denoise_image(model, img_path, IN_CHANS, DEVICE)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as oom_err:
            if "out of memory" in str(oom_err).lower() or isinstance(oom_err, torch.cuda.OutOfMemoryError):
                print(f"  OOM on full image — switching to tiled inference: {fname}")
                torch.cuda.empty_cache()
                out = denoise_tiled(model, img_path, IN_CHANS, TILE_SIZE, TILE_OVERLAP, DEVICE)
            else:
                raise

        save_path = os.path.join(STANDALONE_OUT, Path(fname).stem + ".png")
        Image.fromarray(out, mode="L").save(save_path)

        if (i + 1) % 10 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1:>4}/{len(img_paths)}] saved: {fname}")

    print(f"\\nStandalone inference complete. Results in: {STANDALONE_OUT}")\
"""))

# ── 12. Visual Comparison ─────────────────────────────────────────────────────
inf_cells.append(md("""\
## 12  Visual Side-by-Side Comparison

Shows `VIS_NUM_IMAGES` randomly selected validation images:
**Noisy input → Model output → Ground truth**\
"""))
inf_cells.append(code("""\
import matplotlib.pyplot as plt
import random, os
import numpy as np
from PIL import Image

random.seed(42)
val_fnames = sorted(f for f in os.listdir(LOCAL_VAL_CLEAN) if not f.startswith("."))
idxs = random.sample(range(len(val_fnames)), min(VIS_NUM_IMAGES, len(val_fnames)))

n_rows = len(idxs)
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)

for row, idx in enumerate(idxs):
    fname      = val_fnames[idx]
    noisy_path = os.path.join(LOCAL_VAL_NOISY, fname)
    clean_path = os.path.join(LOCAL_VAL_CLEAN,  fname)

    noisy_arr = np.array(Image.open(noisy_path).convert("L"))
    clean_arr = np.array(Image.open(clean_path).convert("L"))

    # Run inference (tiled fallback if needed)
    try:
        denoised_arr = denoise_image(model, noisy_path, IN_CHANS, DEVICE)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            denoised_arr = denoise_tiled(model, noisy_path, IN_CHANS,
                                         TILE_SIZE, TILE_OVERLAP, DEVICE)
        else:
            raise

    for col, (arr, title) in enumerate([
        (noisy_arr,   "Noisy Input"),
        (denoised_arr,"Denoised (MambaIRv2)"),
        (clean_arr,   "Ground Truth"),
    ]):
        ax = axes[row, col]
        ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

plt.suptitle("MambaIRv2 TPAF Denoising — Validation Samples", fontsize=14, y=1.01)
plt.tight_layout()

comp_path = os.path.join(LOCAL_RESULTS_DIR, "comparison.png")
plt.savefig(comp_path, dpi=100, bbox_inches="tight")
plt.show()
print(f"Comparison figure saved to: {comp_path}")\
"""))

# ── 13. Aggregate Metrics ─────────────────────────────────────────────────────
inf_cells.append(md("""\
## 13  Aggregate Metrics (Phase 8)

Computes PSNR and SSIM for:
- **Noisy vs Clean** — baseline (how much noise there is)
- **Denoised vs Clean** — model performance

Results are also written to a CSV file.\
"""))
inf_cells.append(code("""\
import os, csv
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

val_fnames = sorted(f for f in os.listdir(LOCAL_VAL_CLEAN) if not f.startswith("."))

# We'll denoise each val image on-the-fly for metrics
# (re-uses the already-loaded model)

rows = []
psnrs_noisy, ssims_noisy       = [], []
psnrs_denoised, ssims_denoised = [], []

print(f"Computing metrics on {len(val_fnames)} validation pairs ...")

for i, fname in enumerate(val_fnames):
    clean_arr = np.array(Image.open(os.path.join(LOCAL_VAL_CLEAN, fname)).convert("L"))
    noisy_arr = np.array(Image.open(os.path.join(LOCAL_VAL_NOISY, fname)).convert("L"))

    try:
        den_arr = denoise_image(model, os.path.join(LOCAL_VAL_NOISY, fname),
                                IN_CHANS, DEVICE)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
            den_arr = denoise_tiled(model, os.path.join(LOCAL_VAL_NOISY, fname),
                                    IN_CHANS, TILE_SIZE, TILE_OVERLAP, DEVICE)
        else:
            raise

    pn = calc_psnr(clean_arr, noisy_arr,   data_range=255)
    sn = calc_ssim(clean_arr, noisy_arr,   data_range=255)
    pd = calc_psnr(clean_arr, den_arr,     data_range=255)
    sd = calc_ssim(clean_arr, den_arr,     data_range=255)

    psnrs_noisy.append(pn);   ssims_noisy.append(sn)
    psnrs_denoised.append(pd); ssims_denoised.append(sd)
    rows.append({"filename": fname,
                 "psnr_noisy": pn, "ssim_noisy": sn,
                 "psnr_denoised": pd, "ssim_denoised": sd})

    if (i + 1) % 50 == 0 or (i + 1) == len(val_fnames):
        print(f"  [{i+1}/{len(val_fnames)}]  "
              f"running avg denoised PSNR: {np.mean(psnrs_denoised):.2f} dB")

# Save CSV
csv_path = os.path.join(LOCAL_RESULTS_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Summary table
print()
print(f"{'Metric':<20} {'Noisy Input':>15} {'Denoised':>15}  {'Improvement':>12}")
print("─" * 65)
print(f"{'PSNR (dB)':<20} {np.mean(psnrs_noisy):>15.4f} "
      f"{np.mean(psnrs_denoised):>15.4f}  "
      f"{np.mean(psnrs_denoised)-np.mean(psnrs_noisy):>+11.4f}")
print(f"{'SSIM':<20} {np.mean(ssims_noisy):>15.4f} "
      f"{np.mean(ssims_denoised):>15.4f}  "
      f"{np.mean(ssims_denoised)-np.mean(ssims_noisy):>+11.4f}")
print()
print(f"Per-image metrics saved to: {csv_path}")\
"""))

# ── 14. Copy Results to Drive ─────────────────────────────────────────────────
inf_cells.append(md("""\
## 14  Copy All Results to Google Drive\
"""))
inf_cells.append(code("""\
import shutil, os

shutil.copytree(LOCAL_RESULTS_DIR, DRIVE_RESULTS_DIR, dirs_exist_ok=True)
print(f"Results copied to Drive: {DRIVE_RESULTS_DIR}")
print("Contents:")
for root, dirs, files in os.walk(DRIVE_RESULTS_DIR):
    level = root.replace(DRIVE_RESULTS_DIR, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 2:
        for f in files[:5]:
            print(f"{indent}  {f}")
        if len(files) > 5:
            print(f"{indent}  ... ({len(files) - 5} more files)")\
"""))

# ── Save inference.ipynb ─────────────────────────────────────────────────────
save(notebook(inf_cells), "inference.ipynb")

print("\nDone. Both notebooks generated successfully.")
