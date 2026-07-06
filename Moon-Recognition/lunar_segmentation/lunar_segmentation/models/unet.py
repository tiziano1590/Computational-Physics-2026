import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive 3×3 Conv → BN → ReLU blocks.

    With use_residual=True a shortcut is added after the second BN (before the
    final ReLU), turning the block into a residual unit.  When in_ch ≠ out_ch
    a 1×1 conv+BN projection aligns the shortcut dimensions.

    Why residuals?  They let gradients bypass the convolutions during
    backpropagation, making deeper networks much easier to train.
    """

    def __init__(self, in_ch: int, out_ch: int, use_residual: bool = False):
        super().__init__()
        self.use_residual = use_residual

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        # Second conv has no trailing ReLU — applied after the residual addition
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if use_residual:
            self.skip = (
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                )
                if in_ch != out_ch
                else nn.Identity()
            )
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        if self.use_residual:
            out = out + self.skip(x)
        return self.relu(out)


class Down(nn.Module):
    """MaxPool2d halving + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, use_residual: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, use_residual=use_residual),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Learned 2× upsampling (ConvTranspose2d) + skip-cat + DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, use_residual: bool = False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch, use_residual=use_residual)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to match encoder feature map size (handles odd dimensions)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1×1 conv that maps feature channels to class logits."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SmallUNet(nn.Module):
    """U-Net with 3 down/up stages (4 resolution levels) for multi-label
    pixel-wise segmentation.

    Spatial flow with base_width=32 and the 256×256 training tiles:

        (3, 256,256) → inc   → (32, 256,256)  ← skip 1
                     → down1 → (64, 128,128)   ← skip 2
                     → down2 → (128, 64, 64)   ← skip 3
                     → down3 → (256, 32, 32)   bottleneck [dropout here]
                     → up1   → (128, 64, 64)   + skip 3
                     → up2   → (64, 128,128)   + skip 2
                     → up3   → (32, 256,256)   + skip 1
                     → outc  → (C,  256,256)   logits

    The network is fully convolutional, so any input size divisible by 8
    works; odd intermediate sizes are handled by padding in Up.

    Args:
        in_channels:  input channels (3 for norm+CLAHE+Sobel).
        num_classes:  output segmentation classes.
        base_width:   channel count at the first encoder level.
        use_residual: add ResNet-style skip connections inside every
                      DoubleConv block.  Improves gradient flow;
                      slightly increases parameter count.
        dropout:      Dropout2d probability at the bottleneck (0 = off).
                      Helps regularise when training labels are sparse.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_width: int = 32,
        use_residual: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        w = base_width
        self.inc   = DoubleConv(in_channels, w,     use_residual=use_residual)
        self.down1 = Down(w,     w * 2, use_residual=use_residual)
        self.down2 = Down(w * 2, w * 4, use_residual=use_residual)
        self.down3 = Down(w * 4, w * 8, use_residual=use_residual)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.up1   = Up(w * 8, w * 4, use_residual=use_residual)
        self.up2   = Up(w * 4, w * 2, use_residual=use_residual)
        self.up3   = Up(w * 2, w,     use_residual=use_residual)
        self.outc  = OutConv(w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop(x4)
        x  = self.up1(x4, x3)
        x  = self.up2(x,  x2)
        x  = self.up3(x,  x1)
        return self.outc(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
