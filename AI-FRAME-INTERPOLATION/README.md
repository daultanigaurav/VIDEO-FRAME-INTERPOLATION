# Video Frame Interpolation: GANs-U-Net-Inspired Pipeline

## Project Overview

This project implements a **video frame interpolation system** using a GANs-U-Net-inspired architecture. The current implementation features a **simplified deterministic network** that serves as a foundation for demonstrating the pipeline structure. The project is designed to be transparent about its current state: it uses a **simulation mode** for the adversarial components while maintaining a professional architecture that can be extended to a full GANs-U-Net implementation.

### Current Status

- ✅ **Working**: Simple deterministic interpolation model (end-to-end functional)
- 🔄 **Simulated**: GAN discriminator and adversarial training (placeholder with synthetic losses)
- 📋 **Future Work**: Full GANs-U-Net implementation with real adversarial training

## Architecture

The project follows a GANs-U-Net-inspired design:

```
Input Frames (t, t+1)
    ↓
[U-Net Generator] ← Currently: Simple Interpolator
    ↓
Interpolated Frame (t+0.5)
    ↓
[Discriminator] ← Currently: Simulated (placeholder)
    ↓
Adversarial Loss ← Currently: Synthetic curve for demo
```

### Current Implementation

- **Generator**: Simple U-Net-like architecture (deterministic, no adversarial training)
- **Discriminator**: Placeholder wrapper that simulates discriminator behavior
- **Loss Functions**: 
  - Real: Reconstruction loss (L1/L2)
  - Simulated: Adversarial loss (synthetic curves for visualization)

### Future GANs-U-Net Implementation

When converting to a full GAN:
- Replace simple interpolator with U-Net generator trained adversarially
- Implement real discriminator network
- Add adversarial loss to training loop
- Implement gradient penalty or spectral normalization for stability

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── src/
│   ├── interpolator.py         # Simple working interpolation model
│   ├── simulated_gan_wrapper.py # GAN simulation wrapper (demo mode)
│   └── create_demo_video.py    # Script to generate demo videos
├── notebooks/
│   └── demo_visuals.ipynb      # Visualizations and analysis
├── assets/
│   └── architecture.svg        # Architecture diagram
└── slides/
    └── presentation.md         # Project presentation slides
```

## Installation

1. **Clone or navigate to the project directory**

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start: Generate Interpolated Frames

```bash
python src/create_demo_video.py --input_frames frame1.jpg frame2.jpg --output output_video.mp4
```

### Using the Simple Interpolator

```python
from src.interpolator import SimpleInterpolator

interpolator = SimpleInterpolator()
frame_t = load_image("frame1.jpg")
frame_t1 = load_image("frame2.jpg")

interpolated = interpolator.interpolate(frame_t, frame_t1, alpha=0.5)
save_image(interpolated, "interpolated_frame.jpg")
```

### Using the Simulated GAN Wrapper (Demo Mode)

```python
from src.simulated_gan_wrapper import SimulatedGANWrapper

gan = SimulatedGANWrapper()
result = gan.generate_with_simulation(frame_t, frame_t1)

# Access interpolated frame
interpolated_frame = result['interpolated_frame']

# Access simulated metrics (for demo purposes)
fake_discriminator_loss = result['simulated_discriminator_loss']
fake_generator_loss = result['simulated_generator_loss']
```

## How Simulation Works

The `SimulatedGANWrapper` demonstrates how a full GAN pipeline would work:

1. **Generator Forward Pass**: Uses the simple interpolator to generate frames
2. **Simulated Discriminator**: 
   - Placeholder that doesn't actually classify frames
   - Generates synthetic loss curves based on frame quality metrics (SSIM, PSNR)
   - Saves plots showing "adversarial training" progress (for demo)
3. **Simulated Losses**: 
   - Discriminator loss: Synthetic curve that decreases over "epochs"
   - Generator loss: Synthetic curve showing training progress
   - Both are clearly labeled as simulated in plots

### Transparency Markers

All simulated components are clearly marked:
- Function names: `_simulate_discriminator()`, `_generate_synthetic_losses()`
- Comments: `# SIMULATION: This would be replaced with real discriminator`
- Logs: `[SIMULATION] Generating synthetic discriminator loss...`
- Plot labels: `"Simulated Discriminator Loss (Demo Only)"`

## Limitations

### Current Implementation

1. **No Real Adversarial Training**: The discriminator is a placeholder
2. **Deterministic Output**: No adversarial feedback to improve generator
3. **Synthetic Losses**: Loss curves are generated, not from actual training
4. **Limited Generalization**: Simple model may not handle complex motion well

### What's Missing for Full GAN

1. **Real Discriminator Network**: CNN that classifies real vs. interpolated frames
2. **Adversarial Training Loop**: Alternating generator/discriminator updates
3. **Advanced Losses**: 
   - Perceptual loss (VGG features)
   - Adversarial loss (from discriminator)
   - Temporal consistency loss
4. **Training Infrastructure**: 
   - Large-scale dataset
   - Multi-GPU training
   - Checkpointing and resuming

## Next Steps: Implementing Full GANs-U-Net

### Phase 1: U-Net Generator Architecture
- [ ] Implement full U-Net with skip connections
- [ ] Add residual blocks for better gradient flow
- [ ] Implement multi-scale feature extraction

### Phase 2: Discriminator Network
- [ ] Design PatchGAN discriminator
- [ ] Implement spectral normalization for stability
- [ ] Add gradient penalty (WGAN-GP) or other regularization

### Phase 3: Adversarial Training
- [ ] Implement alternating training loop
- [ ] Add perceptual loss using pre-trained VGG
- [ ] Implement temporal consistency loss
- [ ] Add learning rate scheduling

### Phase 4: Optimization
- [ ] Multi-GPU training support
- [ ] Mixed precision training (FP16)
- [ ] Advanced data augmentation
- [ ] Hyperparameter tuning

### Code Migration Guide

To convert from simulation to real GAN:

1. **Replace Simple Interpolator**:
   ```python
   # Current (simulation)
   from src.interpolator import SimpleInterpolator
   
   # Future (real GAN)
   from src.gan_generator import UNetGenerator
   generator = UNetGenerator()
   ```

2. **Replace Simulated Discriminator**:
   ```python
   # Current (simulation)
   fake_loss = self._simulate_discriminator(frame)
   
   # Future (real GAN)
   from src.gan_discriminator import PatchGANDiscriminator
   discriminator = PatchGANDiscriminator()
   real_loss = discriminator(real_frames)
   fake_loss = discriminator(generated_frames)
   ```

3. **Replace Synthetic Losses**:
   ```python
   # Current (simulation)
   gen_loss, disc_loss = self._generate_synthetic_losses()
   
   # Future (real GAN)
   gen_loss = adversarial_loss + perceptual_loss + l1_loss
   disc_loss = real_loss - fake_loss + gradient_penalty
   ```

## Demo and Visualization

Run the Jupyter notebook for interactive visualizations:

```bash
jupyter notebook notebooks/demo_visuals.ipynb
```

The notebook includes:
- Input frame visualization
- Interpolated frame comparison
- Simulated adversarial loss curves
- Quality metrics (SSIM, PSNR)

## Presentation

See `slides/presentation.md` for a 5-slide presentation covering:
1. Problem statement
2. GANs-U-Net approach
3. Current demo implementation
4. Limitations and transparency
5. Next steps for full implementation

## References

- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- **GANs**: Goodfellow et al., "Generative Adversarial Networks"
- **Frame Interpolation**: Various papers on video frame interpolation with GANs

## License

[Specify your license here]

## Contact

[Your contact information]
