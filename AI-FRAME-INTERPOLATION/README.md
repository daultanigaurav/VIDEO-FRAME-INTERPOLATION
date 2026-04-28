# VFI Project

Production-style frame interpolation project with Streamlit UI, deterministic fallback safety, and robust SSIM/PSNR reporting.

## Run

```bash
streamlit run app/app.py
```

## Setup

1. Create virtual environment
2. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For dev/notebooks:

```bash
pip install -r requirements-dev.txt
```

## Project Tree

```text
VFI-Project/
├── app/                    # Streamlit app entrypoint
│   └── app.py
├── core/                   # Core interpolation, metrics, video utilities
│   ├── interpolate.py
│   ├── model_loader.py
│   ├── video_utils.py
│   ├── metrics.py
│   └── utils.py
├── models/                 # Trained model weights (optional)
├── data/                   # Sample input assets
│   ├── sample_frames/
│   └── sample_videos/
├── outputs/                # Generated artifacts (ignored in git)
│   ├── videos/
│   ├── frames/
│   └── temp/
├── assets/                 # Static assets (architecture diagrams, etc.)
├── notebooks/              # Experiment notebooks
├── tests/                  # Automated tests
├── docs/                   # Project documentation
│   ├── PROJECT_OVERVIEW.md
│   └── PROJECT_STRUCTURE.md
├── scripts/                # Utility scripts (training entrypoint)
│   └── train.py
├── config/                 # Runtime configuration
│   └── config.yaml
├── requirements.txt
├── requirements-dev.txt
├── README.md
├── .gitignore
└── run.bat
```

## Notes

- **Fallback mode**: if `models/trained_model.pth` is missing/unloadable, interpolation falls back to deterministic blend-based generation with a UI warning.
- **GAN simulation**: legacy GAN simulation code under `src/` is demo-oriented and not part of the main production runtime path.
