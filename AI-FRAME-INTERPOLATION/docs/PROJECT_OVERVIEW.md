# Project Overview: AI Frame Interpolation System (Current State)

## 1) What this project does today

This project is a working Streamlit application for video frame interpolation.

You can:
- upload **two images** or a **video file**
- generate intermediate frames
- export an output video
- inspect quality metrics (SSIM and PSNR) and optical-flow visualization

The practical runtime path is in `app/app.py` + `core/*`.  
The separate `src/*` GAN path is demo/simulation oriented.

## 2) Current architecture and behavior

### App/UI layer
- `app/app.py`
  - Streamlit entrypoint.
  - Handles input, settings, generation, download flow.
  - Displays quality charts and metrics.

### Interpolation pipeline
- `core/interpolate.py`
  - Orchestrates frame/video interpolation, output writes, and runtime metrics.
  - Uses `outputs/frames` and `outputs/videos` for generated artifacts.

### Model loading and fallback mode
- `core/model_loader.py`
  - Loads `models/trained_model.pth` when available.
  - If missing/unloadable, enables deterministic blend-based fallback and emits UI warning.
  - Avoids unstable random-weight outputs.

### Metrics behavior
- `core/metrics.py` + `app.py` metrics tab
  - SSIM/PSNR are always displayed after interpolation.
  - If exact metric computation fails, approximate fallback values are used.
  - Clean plotly chart is always rendered.

## 3) Runtime maturity

Status: **functional prototype with production-style folder organization**.

Strengths:
- Reproducible pinned dependencies.
- Deterministic fallback for missing model weights.
- Robust metric rendering behavior.
- Runnable with `streamlit run app/app.py`.

Limitations:
- Quality still depends on trained weights.
- GAN path under `src/` is simulation-oriented, not production adversarial training.

## 4) Recommended next steps

1. Add a validated default `models/trained_model.pth`.
2. Add integration tests for end-to-end app run + metrics rendering.
3. Containerize with Docker and CI checks.
4. Decide long-term path: harden deterministic path vs full GAN stack.
