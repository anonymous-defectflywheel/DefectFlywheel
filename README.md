# DefectFlywheel

DefectFlywheel is a few-shot fabric anomaly detection framework. It builds on One-for-All FSAD and improves the detector with a closed training loop: detector heatmaps mine hard normal regions, targeted pseudo defects are generated in those regions, and the detector is fine-tuned with the resulting hard negatives.

This repository provides the reproduction package for the DefectFlywheel experiments on:

- **WFDD**: [Kaggle: The Woven Fabric Defect Detection Dataset](https://www.kaggle.com/datasets/hodinhtrieu/the-woven-fabric-defect-detection-wfdd)
- **ZJU-Leaper** / `ZJU-Leaper-AllPatterns-MVTec` (`zjuall`): [official website](http://www.qaas.zju.edu.cn/zju-leaper/)

Default reproduction setting:

```text
shot = 2
seed = 4
macro_epochs = 10
context_backend = blipdiffusion_qformer
mine_method = top_k
mine_top_k_ratio = 0.10
eval_policy = final_only
```

---

## 1. Environment

Use one conda environment for the whole pipeline; no separate `anostyler` environment is required.

```bash
conda create -n defect_flywheel python=3.10 -y
conda activate defect_flywheel
pip install --upgrade pip
pip install -r requirements.txt
```

If your CUDA/PyTorch stack needs a specific wheel, install PyTorch first from the official PyTorch selector, then run `pip install -r requirements.txt`.

Quick import check:

```bash
python -c "import torch, torchvision, cv2, diffusers, transformers, timm, kornia, clip; print('ok', torch.__version__, torch.cuda.is_available())"
```

---

## 2. Data and weights

### 2.1 Datasets

Download the datasets from the official sources above, then arrange them in MVTec-style format:

```text
DefectFlywheel/
  datasets/
    WFDD/
      <class>/train/good/...
      <class>/test/<defect>/...
      <class>/ground_truth/<defect>/...
    ZJU-Leaper-AllPatterns-MVTec/
      p_id1/train/good/...
      p_id1/test/defective/...
      p_id1/ground_truth/defective/...
      ...
```

If your datasets are stored elsewhere, pass `DATA_ROOT=/path/to/datasets` when running the scripts.

### 2.2 BLIP-Diffusion weights

DefectFlywheel uses **BLIP-Diffusion Q-Former context** for the default reproduction scripts. It does not use the older BLIP image-captioning context path.

Recommended local layout:

```text
DefectFlywheel/
  pretrained/
    blipdiffusion_model/
      model_index.json              # or a HuggingFace snapshot subdirectory containing model_index.json
      ...
```

Download options:

1. **Prepared artifact folder**: [Google Drive weights and artifacts](https://drive.google.com/drive/folders/1siLVz5BB1-hQxgCrTIPBJir-eovjf01V?usp=drive_link), then place/extract BLIP-Diffusion under `pretrained/blipdiffusion_model/`.
2. **Official Hugging Face model**: [`Salesforce/blipdiffusion`](https://huggingface.co/Salesforce/blipdiffusion), or use the helper script:

```bash
python scripts/download_blipdiffusion.py --output_dir pretrained/blipdiffusion_model
```

Verify the downloaded model:

```bash
find pretrained/blipdiffusion_model -name model_index.json | head
```

If weights are stored elsewhere, pass `PRETRAINED_DIR=/path/to/pretrained`.

---

## 3. Pre-run check

From the repository root:

```bash
ls datasets/WFDD
ls datasets/ZJU-Leaper-AllPatterns-MVTec
find pretrained/blipdiffusion_model -name model_index.json | head
python scripts/run_experiments.py --help | grep context_backend
```

The help output should include:

```text
--context_backend {blipdiffusion_qformer,blip_caption,static_prompt}
```

---

## 4. Reproduce DefectFlywheel

### WFDD

```bash
GPU_ID=0 bash scripts/reproduce_defectflywheel_wfdd.sh
```

### ZJU-Leaper (`zjuall`)

```bash
GPU_ID=0 bash scripts/reproduce_defectflywheel_zjuall.sh
```

### Custom paths

```bash
GPU_ID=0 \
DATA_ROOT=/path/to/datasets \
PRETRAINED_DIR=/path/to/pretrained \
EXP_ROOT=/path/to/experiments \
bash scripts/reproduce_defectflywheel_wfdd.sh
```

Replace the last line with `bash scripts/reproduce_defectflywheel_zjuall.sh` for ZJU-Leaper.

### Smoke test

For a quick pipeline check, use one macro epoch:

```bash
GPU_ID=0 EPOCHS=1 bash scripts/reproduce_defectflywheel_wfdd.sh
```

---

## 5. Outputs

Each script creates a timestamped run directory:

```text
experiments/reproduce_defectflywheel_<dataset>_2shot_seed42_10epoch_<yyyymmdd_HHMMSS>_gpu<id>/
```

Core files:

```text
command.sh
run_config.json
stdout_stderr.log
env_snapshot.txt
pip_freeze.txt
nvidia_smi_before.txt
nvidia_smi_after.txt
raw_metrics.json
metrics_summary.csv
metrics_summary.jsonl
checkpoint_manifest.json
method_outputs/
```

Paper-aligned results use the final DefectFlywheel round only because the scripts set `--eval_policy final_only` and `--baseline_eval_policy final_only`.

---

## 6. Runtime options

| Variable | Default | Description |
|---|---|---|
| `GPU_ID` | `0` | CUDA device id used through `CUDA_VISIBLE_DEVICES` |
| `DATA_ROOT` | `<repo>/datasets` | Parent directory containing `WFDD` and `ZJU-Leaper-AllPatterns-MVTec` |
| `PRETRAINED_DIR` | `<repo>/pretrained` | Directory containing `blipdiffusion_model/` |
| `EXP_ROOT` | `<repo>/experiments` | Output directory for timestamped runs |
| `SHOT` | `2` | Few-shot support count |
| `SEED` | `4` | Random seed |
| `EPOCHS` | `10` | Macro epochs; paper-aligned reproduction default is 10 |

Keep the defaults for paper-aligned reproduction unless running a smoke test.

---

## 7. Repository hygiene

Datasets, pretrained weights, checkpoints, and experiment outputs are intentionally ignored by Git. Do not upload large data or model files to GitHub; share them through the Google Drive folder or another artifact host.
