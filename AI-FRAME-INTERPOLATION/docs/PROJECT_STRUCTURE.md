# Project Structure

```text
VFI-Project/
├── app/                    # Streamlit app entrypoint
│   └── app.py
├── core/                   # Core interpolation + metrics + utils
│   ├── interpolate.py
│   ├── model_loader.py
│   ├── video_utils.py
│   ├── metrics.py
│   └── utils.py
├── models/                 # Trained model checkpoints
├── data/                   # Sample inputs / datasets
│   ├── sample_frames/
│   └── sample_videos/
├── outputs/                # Generated outputs (ignored by git)
│   ├── videos/
│   ├── frames/
│   └── temp/
├── assets/                 # Static assets and diagrams
├── notebooks/              # Research and visual experiments
├── tests/                  # Unit/integration tests
├── docs/                   # Project docs
│   ├── PROJECT_OVERVIEW.md
│   └── PROJECT_STRUCTURE.md
├── scripts/                # Script entrypoints (training, utilities)
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

- Primary run command: `streamlit run app/app.py`
- Main runtime path: `app/` + `core/`
- `src/` (if present) is legacy/demo simulation code and not required for main app execution.
