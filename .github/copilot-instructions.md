# GenDataCode - Copilot Instructions

## Quick Start

**Core structure:**
- `model/`: 5 independent generative models (Diffusion-TS, PaD-TS, TimeGAN, TimeVAE, CTD_Mamba_Diff)
- `dataProcess/`: per-dataset preprocessing pipelines
- `configs/`: dataset-specific hyperparameters, loaded via `configuration.py`

**Run examples:**
```bash
python model/CTD_Mamba_Diff/run.py
python model/PaD-TS-main/run.py -d FD001 -window 24 -steps 0
python model/Diffusion-TS/run_diffusion_ts.py --mode train|sample|infill
python model/TimeVAE/run_time_vae.py --dataset ETTh1 --model vae_type
python model/TimeGAN/run_time_gan.py --dataset Traffic
```

**Datasets:** `ETTh1`, `ETTh2`, `AirQuality(bj)`, `AirQuality(Italian)`, `Traffic`, `FD001`, `FD002`, `FD003`, `FD004`

**Special handling:**
- **Traffic**: Use `last_col_only=True` → model processes single feature channel only.
- **Conditions**: `SeqConditionVAE` encodes + expands via `.unsqueeze(1).repeat(1, L, 1)`.
  - Dim mapping: ETT=6, C-MAPSS=1, AirQuality=1, Traffic=11.

## Key Architectural Patterns

| Component | Pattern | Location |
|-----------|---------|----------|
| **Config loading** | Parse by dataset name; support FD001-004, ETT*, Air*, Traffic | `configuration.py` |
| **Mamba diffusion** | FFT extraction → cond concat → Mamba block → noise predictor → reverse schedule | `model/CTD_Mamba_Diff/lib/ctd_mamba_diff.py` |
| **Data shapes** | Input `[B,L,F]` → Conv1d expects `[B,C,L]` (transpose at `x.transpose(1,2)`) | `model/TimeVAE/lib/vae_conv_model.py` |
| **Training loss** | MSE on `eps_theta` (noise prediction); clip gradients at norm 1.0 | `model/Diffusion-TS/engine/solver.py#L145` |
| **Batch generation** | Split into chunks (8 samples); call `torch.cuda.empty_cache()` after each batch | `model/CTD_Mamba_Diff/modules/model.py#L511` |
| **Output validation** | Replace NaN/Inf; clamp to `[-3.0, 3.0]` after generation | `model/CTD_Mamba_Diff/modules/model.py#L511-L512` |

## Debugging Checklist

### D1: Config Not Found or Mismatched Dataset

**Symptom:** `FileNotFoundError`, config section not found  
**Check:**
1. [configuration.py](configuration.py#L1-L11) - Verify dataset name matches expected list and config file exists at `configs/{DATASET}/{DATASET}_{MODEL}.conf`
2. [model/CTD_Mamba_Diff/lib/config_loader.py](model/CTD_Mamba_Diff/lib/config_loader.py#L48-L49) - Reading keys like `early_stop_patience`, `early_stop_delta` (lines 48-49)
3. [model/TimeGAN/config_loader.py](model/TimeGAN/config_loader.py#L88-L91) - If GPU fails, device fallback to CPU (lines 88-91)

**Action:** Check config file format, verify all required sections exist (`[Training]`, `[System]`, etc.)

---

### D2: CUDA Out-of-Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`  
**Check:**
1. [model/CTD_Mamba_Diff/lib/config_loader.py](model/CTD_Mamba_Diff/lib/config_loader.py#L71-L72) - Verify GPU setup and device ID (lines 71-72)
2. [model/PaD-TS-main/training.py](model/PaD-TS-main/training.py#L36-L39) - Confirm model moved to device (lines 36-39)
3. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L511) - GPU cache cleanup in generation loop (line 511+) **✅ FIXED**
4. [model/PaD-TS-main/eval_utils/MMD.py](model/PaD-TS-main/eval_utils/MMD.py#L123-L132) - Explicit cleanup after batch ops (lines 123-132)

**Action:** Reduce batch size in config file, or move conditions to CPU with `.cpu()` before passing to model. **Note**: CTD_Mamba_Diff now clears cache every 50 steps.

---

### D3: Tensor Shape Mismatch

**Symptom:** `RuntimeError: conv1d expects 3D input`, shape broadcast errors  
**Check:**
1. [model/TimeVAE/lib/vae_conv_model.py](model/TimeVAE/lib/vae_conv_model.py#L31-L34) - Transpose before Conv1d: `[B,L,F]` → `[B,F,L]` (lines 31-34)
2. [model/TimeVAE/lib/vae_dense_model.py](model/TimeVAE/lib/vae_dense_model.py#L54-L55) - Reshape output dimensions (lines 54-55)
3. [model/CTD_Mamba_Diff/modules/tsp_encoder.py](model/CTD_Mamba_Diff/modules/tsp_encoder.py#L14-L15) - **✅ FIXED**: AvgPool1d/MaxPool1d now use `padding=kernel_size//2` to preserve shape (lines 14-15)
4. [TSlib/lib/dataloader.py](TSlib/lib/dataloader.py#L42-L47) - Handle 2D/3D shape normalization (lines 42-47)
5. [TSlib/lib/dataloader.py](TSlib/lib/dataloader.py#L56-L58) - Restore original shape after processing (lines 56-58)

**Action:** Verify input tensor is `[B,L,F]` before passing to model; check condition expansion `.repeat(1, L, 1)`. TSPEncoder now guarantees consistent shapes across ET/ES/EP.

---

### D4: Dataset File Not Found

**Symptom:** `FileNotFoundError: ...data file not found`, assertion error on dataset name  
**Check:**
1. [TSlib/lib/dataloader.py](TSlib/lib/dataloader.py#L224-L240) - Validate dataset name against allowed list (lines 224+); use exact capitalization
2. [TSlib/lib/dataloader.py](TSlib/lib/dataloader.py#L162-L168) - Check data folder and file existence explicitly (lines 162-168)
3. [model/CTD_Mamba_Diff/run.py](model/CTD_Mamba_Diff/run.py#L33-L48) - Config file per dataset match (lines 33-45)

**Action:** Verify `data/` folder structure matches dataset name (e.g., `data/etth1/`, `data/FD001/`); check file permissions.

---

### D5: Training Loss is NaN or Stuck

**Symptom:** Loss becomes NaN after first epoch; loss plateaus with no improvement  
**Check:**
1. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L354) - Check loss improvement delta threshold (line 354)
2. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L217-L218) - Verify early stopping patience and delta from config (lines 217-218)
3. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L383-L384) - Early stopping break condition (lines 383-384)
4. [model/Diffusion-TS/engine/solver.py](model/Diffusion-TS/engine/solver.py#L145) - Gradient clipping applied (line 145)

**Action:** Check log for NaN count; if present, inspect `torch.nan_to_num()` or clamp logic (lines 511-512 in CTD_Mamba). Reduce learning rate in config. **Note**: Cosine schedule no longer distorts with improper clamping.

---

### D6: Generated Data Contains NaN/Inf

**Symptom:** Output synthetic data has NaN or very large values  
**Check:**
1. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L517) - Log NaN count in output (line 517)
2. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L511-L512) - Verify NaN replacement and clamping applied (lines 511-512)
3. [model/PaD-TS-main/eval_utils/MMD.py](model/PaD-TS-main/eval_utils/MMD.py#L27) - Check metric NaN handling (line 27)

**Action:** Inspect diffusion reverse schedule parameters; if persists, post-process output with `.nan_to_num()` and clamp manually.

---

### D7: Device Configuration Errors (CTD_Mamba_Diff specific) **[NEW]**

**Symptom:** Model always uses GPU even when `device='cpu'` is set, or "Expected all tensors on same device" errors  
**Check:**
1. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L32-L37) - Device logic now correctly handles CPU forcing (lines 32-37) **✅ FIXED**
2. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L107) - VAE condition moved to correct device (line 107) **✅ FIXED**

**Action:** Ensure config has `device=cpu` or `device=cuda:0`. If errors persist after fixes, verify all conditional tensors call `.to(self.device)`.

---

### D8: Condition Data Mismatch (CTD_Mamba_Diff specific) **[NEW]**

**Symptom:** Training fails silently with wrong data mapping, or `ValueError: condition count != data count`  
**Check:**
1. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L98-L101) - Input validation checks feature dimension (lines 98-101) **✅ FIXED**
2. [model/CTD_Mamba_Diff/modules/model.py](model/CTD_Mamba_Diff/modules/model.py#L275-L282) - Condition validation no longer silent-truncates (lines 275-282) **✅ FIXED**

**Action:** Verify condition shape matches `[num_samples, ...]`. If ValueError raised, ensure conditions and data are aligned before training.

---

**Workflow priority:** Config → Dataset → Device/CUDA → Shapes → Training loss → Generation quality.

> For model-specific questions (e.g., TimeVAE encoder quirk), search within the model's directory first; patterns often differ across implementations.

## CTD_Mamba_Diff Critical Fixes (Applied 2026-03-24)

⚠️ **8 critical issues fixed** - See [model/CTD_Mamba_Diff/DETAILED_FIXES_REPORT.md](model/CTD_Mamba_Diff/DETAILED_FIXES_REPORT.md) for full details.

Key improvements:
- ✅ GPU OOM fixed (cache cleaning every 50 diffusion steps)
- ✅ TSPEncoder shape consistency (ET/ES/EP now [B,L,F] uniform)
- ✅ Device logic corrected (CPU forcing now works)
- ✅ Cosine schedule no longer distorted by improper clamping
- ✅ Input validation added (clear error messages)
- ✅ Condition data alignment enforced (no silent truncation)

