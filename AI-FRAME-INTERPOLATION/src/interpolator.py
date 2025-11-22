import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class SimpleInterpolator(nn.Module):
    """
    Simple deterministic interpolation model.
    
    This is a working, end-to-end functional model that serves as the foundation.
    In a full GANs-U-Net implementation, this would be replaced with a U-Net generator
    trained adversarially.
    
    Architecture: Simple encoder-decoder with skip connections (U-Net-like structure)
    """
    
    def __init__(self, in_channels: int = 6, out_channels: int = 3):
        super(SimpleInterpolator, self).__init__()
        
        self.encoder1 = self._conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.encoder2 = self._conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.encoder3 = self._conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.bottleneck = self._conv_block(256, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = self._conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = self._conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = self._conv_block(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, 1)
        self.activation = nn.Tanh()
    
    def _conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        enc3 = self.encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)
        
        bottleneck = self.bottleneck(enc3_pool)
        
        dec3 = self.up3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.final(dec1)
        output = self.activation(output)
        
        return output
    
    def interpolate(
        self, 
        frame1: np.ndarray, 
        frame2: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Interpolate between two frames.
        
        Args:
            frame1: First frame (H, W, 3) in [0, 255]
            frame2: Second frame (H, W, 3) in [0, 255]
            alpha: Interpolation factor (0.0 = frame1, 1.0 = frame2)
        
        Returns:
            Interpolated frame (H, W, 3) in [0, 255]
        """
        self.eval()
        
        if frame1.shape != frame2.shape:
            h, w = min(frame1.shape[0], frame2.shape[0]), min(frame1.shape[1], frame2.shape[1])
            frame1 = frame1[:h, :w]
            frame2 = frame2[:h, :w]
        
        h, w = frame1.shape[:2]
        
        if h % 16 != 0 or w % 16 != 0:
            new_h = ((h // 16) + 1) * 16
            new_w = ((w // 16) + 1) * 16
            frame1 = cv2.resize(frame1, (new_w, new_h))
            frame2 = cv2.resize(frame2, (new_w, new_h))
        
        frame1_norm = frame1.astype(np.float32) / 127.5 - 1.0
        frame2_norm = frame2.astype(np.float32) / 127.5 - 1.0
        
        frame1_tensor = torch.from_numpy(frame1_norm).permute(2, 0, 1).float().unsqueeze(0)
        frame2_tensor = torch.from_numpy(frame2_norm).permute(2, 0, 1).float().unsqueeze(0)
        
        input_tensor = torch.cat([frame1_tensor, frame2_tensor], dim=1)
        
        device = next(self.parameters()).device
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            output = self.forward(input_tensor)
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        output_np = (output_np + 1.0) * 127.5
        output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        
        if output_np.shape[:2] != (h, w):
            output_np = cv2.resize(output_np, (w, h))
        
        return output_np


def load_interpolator(checkpoint_path: Optional[str] = None, device: str = 'cpu') -> SimpleInterpolator:
    """
    Load the interpolation model.
    
    Args:
        checkpoint_path: Path to model checkpoint (if None, returns untrained model)
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    model = SimpleInterpolator()
    model = model.to(device)
    
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")
            print("Using randomly initialized weights")
    else:
        print("Using randomly initialized weights (no checkpoint provided)")
    
    return model

