"""
Script to create demo videos from interpolated frames.

This script runs the interpolation pipeline and assembles output videos.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import List, Optional

from interpolator import SimpleInterpolator, load_interpolator
from simulated_gan_wrapper import SimulatedGANWrapper


def load_image(path: str) -> np.ndarray:
    """Load an image from file."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return img


def save_image(image: np.ndarray, path: str):
    """Save an image to file."""
    cv2.imwrite(path, image)


def create_video_from_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30
):
    """
    Create a video from a list of frames.
    
    Args:
        frames: List of frames (numpy arrays)
        output_path: Output video path
        fps: Frames per second
    """
    if not frames:
        raise ValueError("No frames provided")
    
    height, width = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        out.write(frame)
    
    out.release()
    print(f"Saved video to {output_path}")


def interpolate_between_frames(
    frame1: np.ndarray,
    frame2: np.ndarray,
    num_interpolations: int,
    interpolator: SimpleInterpolator
) -> List[np.ndarray]:
    """
    Generate interpolated frames between two frames.
    
    Args:
        frame1: First frame
        frame2: Second frame
        num_interpolations: Number of intermediate frames
        interpolator: Interpolation model
    
    Returns:
        List of frames including originals and interpolated
    """
    frames = [frame1.copy()]
    
    for i in range(1, num_interpolations + 1):
        alpha = i / (num_interpolations + 1)
        print(f"  Generating frame {i}/{num_interpolations} (alpha={alpha:.3f})...")
        
        interpolated = interpolator.interpolate(frame1, frame2, alpha)
        frames.append(interpolated)
    
    frames.append(frame2.copy())
    
    return frames


def main():
    parser = argparse.ArgumentParser(
        description='Create demo video from interpolated frames'
    )
    parser.add_argument(
        '--input_frames',
        nargs=2,
        required=True,
        help='Paths to two input frames'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output_video.mp4',
        help='Output video path'
    )
    parser.add_argument(
        '--num_interpolations',
        type=int,
        default=5,
        help='Number of interpolated frames between input frames'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Output video FPS'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (optional)'
    )
    parser.add_argument(
        '--use_simulation',
        action='store_true',
        help='Use simulated GAN wrapper (for demo)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use (cpu/cuda)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video Frame Interpolation Demo")
    print("=" * 60)
    
    frame1_path, frame2_path = args.input_frames
    
    print(f"\nLoading frames...")
    print(f"  Frame 1: {frame1_path}")
    print(f"  Frame 2: {frame2_path}")
    
    frame1 = load_image(frame1_path)
    frame2 = load_image(frame2_path)
    
    print(f"  Frame 1 shape: {frame1.shape}")
    print(f"  Frame 2 shape: {frame2.shape}")
    
    if args.use_simulation:
        print(f"\n[SIMULATION MODE] Using simulated GAN wrapper...")
        print(f"[SIMULATION MODE] Discriminator outputs will be simulated")
        
        gan_wrapper = SimulatedGANWrapper(
            generator_checkpoint=args.checkpoint,
            device=args.device
        )
        
        print(f"\nGenerating {args.num_interpolations} interpolated frames...")
        result = gan_wrapper.run_simulation_demo(
            frame1, frame2, args.num_interpolations
        )
        
        frames = [result['frame1']]
        for r in result['frames']:
            frames.append(r['frame'])
        frames.append(result['frame2'])
        
        print(f"\n[SIMULATION MODE] Note: All discriminator metrics are simulated")
    else:
        print(f"\nUsing simple interpolator...")
        interpolator = load_interpolator(args.checkpoint, device=args.device)
        
        print(f"\nGenerating {args.num_interpolations} interpolated frames...")
        frames = interpolate_between_frames(
            frame1, frame2, args.num_interpolations, interpolator
        )
    
    print(f"\nAssembling video...")
    print(f"  Total frames: {len(frames)}")
    print(f"  FPS: {args.fps}")
    print(f"  Output: {args.output}")
    
    create_video_from_frames(frames, args.output, args.fps)
    
    print(f"\n{'=' * 60}")
    print("Done!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()

