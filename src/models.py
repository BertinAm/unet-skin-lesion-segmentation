"""
Model factory.

Three architectures for the required comparison:
  * "unet"            -> vanilla U-Net (baseline)            via segmentation-models-pytorch
  * "unetpp"          -> U-Net++ (nested dense skips)        via segmentation-models-pytorch
  * "attention_unet"  -> Attention U-Net (attention gates)   custom implementation below

Using SMP for U-Net / U-Net++ gives us ImageNet-pretrained encoders for free,
which is the industry-standard way to get strong results on small medical sets.
The Attention U-Net is implemented from scratch so the report can discuss the
attention-gate mathematics directly.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# =============================================================================
# Attention U-Net (custom)
# =============================================================================
class ConvBlock(nn.Module):
    """Two 3x3 conv -> BN -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate (Oktay et al., 2018).

    Combines the gating signal `g` (coarse, from the decoder) with the skip
    features `x` (fine, from the encoder) to produce attention coefficients in
    [0,1] that re-weight `x`, suppressing irrelevant background activations.

        q   = ReLU(W_g g + W_x x)
        psi = sigmoid(W_psi q)        # attention coefficients
        out = x * psi
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=True),
                                 nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_x, f_int, 1, bias=True),
                                 nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=True),
                                 nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        attn = self.relu(self.W_g(g) + self.W_x(x))
        attn = self.psi(attn)
        return x * attn


class AttentionUNet(nn.Module):
    """Attention U-Net for binary segmentation (trained from scratch)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 features=(64, 128, 256, 512)):
        super().__init__()
        f1, f2, f3, f4 = features
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(in_channels, f1)
        self.enc2 = ConvBlock(f1, f2)
        self.enc3 = ConvBlock(f2, f3)
        self.enc4 = ConvBlock(f3, f4)
        self.bottleneck = ConvBlock(f4, f4 * 2)

        # Decoder with attention gates
        self.up4 = nn.ConvTranspose2d(f4 * 2, f4, 2, stride=2)
        self.att4 = AttentionGate(f4, f4, f4 // 2)
        self.dec4 = ConvBlock(f4 * 2, f4)

        self.up3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.att3 = AttentionGate(f3, f3, f3 // 2)
        self.dec3 = ConvBlock(f3 * 2, f3)

        self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.att2 = AttentionGate(f2, f2, f2 // 2)
        self.dec2 = ConvBlock(f2 * 2, f2)

        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.att1 = AttentionGate(f1, f1, f1 // 2)
        self.dec1 = ConvBlock(f1 * 2, f1)

        self.final = nn.Conv2d(f1, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self.dec4(torch.cat([self.att4(d4, e4), d4], dim=1))
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(d3, e3), d3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(d2, e2), d2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.att1(d1, e1), d1], dim=1))

        return self.final(d1)   # raw logits (no sigmoid — loss applies it)


# =============================================================================
# Factory
# =============================================================================
def build_model(cfg) -> nn.Module:
    """Instantiate a model from the config `model` block."""
    m = cfg["model"]
    arch = m["arch"].lower()

    if arch == "unet":
        return smp.Unet(
            encoder_name=m["encoder"],
            encoder_weights=m["encoder_weights"],
            in_channels=m["in_channels"],
            classes=m["num_classes"],
        )
    if arch in ("unetpp", "unetplusplus", "unet++"):
        return smp.UnetPlusPlus(
            encoder_name=m["encoder"],
            encoder_weights=m["encoder_weights"],
            in_channels=m["in_channels"],
            classes=m["num_classes"],
        )
    if arch in ("attention_unet", "attunet"):
        return AttentionUNet(
            in_channels=m["in_channels"],
            num_classes=m["num_classes"],
        )
    raise ValueError(f"Unknown arch '{arch}'. Use unet | unetpp | attention_unet.")
