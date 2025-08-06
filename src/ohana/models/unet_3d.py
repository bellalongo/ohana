import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """
        Applies two consecutive 3D convolutions, each followed by batch normalization and ReLU activation
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
            Arguments:
                in_channel (int): number if input channels
                out_channels (int): number of output channels
                mid_channels (int, optional): number of channels in the first convolution 
                    default: out_channels
            Attributes:
                double_conv (nn.Sequential): the sequential block containing the
                    Conv3d -> BN -> ReLU -> Conv3D -> BN -> ReLU
        """
        # Create the init
        super().__init__()

        # Set mid channels to the number of out channels if not already set
        if not mid_channels:
            mid_channels = out_channels

        # 3x3x3 kernels with padding = 1 keep T, H, W unchanged
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
            Run the 2 Conv3D-BatchNorm3D-ReLU blocks, perseving th 3x3x3 kernel, only changing
            the channel dimenstion
            Arguments:
                x (torch.Tensor): input shape of (N, C_in, T, H, W)
            Returns:
                torch.Tensor: output of shape (N, C_out, T, H, W)
        """
        return self.double_conv(x)

class Down3D(nn.Module):
    """
        Downscaling with 3D max-pooling followed by DoubleConv3D, reducing the resolution
        by a factor of 2 in time and space (T, H, W), then increasing the channel capacity
        via convolution
    """
    def __init__(self, in_channels, out_channels):
        """
            Arguments: 
                in_channels (int): number of input channels
                out_channels(int): number of output channels
            Attributes:
                maxpool_conv (nn.Sequential): MaxPool3D(2) -> DoubleConv3D
        """
        # Initalize init
        super().__init__()

        # Downsampling block which has halves (T, H, W) with 3D max-pooling, then applying Conv3D layers
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        """
            Downsample by 2 in (T, H, W), then apply the convolutions
            Arguments:
                x (torch.Tensor): input tensor (N, C_in, T, H, W)
            Returns:
                torch.Tensor: output (N, C_out, T/2, H/2, W/2)
        """
        return self.maxpool_conv(x)

class Up3D(nn.Module):
    """
        Decoder upsamples a low-resolution feature map, aligns it with the corresponding
        encoder feature (skip connection), concats them along channels, and refines with two convolutions
    """
    def __init__(self, in_channels, out_channels, trilinear=True):
        """
            Arguments:
                in_channels (int): channels AFTER concat (C_skip + C_up)
                out_channels (int): channels after the convolution block
                trilinear (bool, optional): if True, uses parameter-free trilinear interpoliation for
                    upsampling, if False, uses ConvTranspose3D
                    default: True
            Notes:
                * interpoliation is lighter (no params) and avoids some artifacts
                * transpsed conv is learnable and may produce a checkboard pattern ??? if not properly 
                configed
        """
        # Create initialization
        super().__init__()

        # Check if trilinear is set to True
        if trilinear:
            # Double the T, H, W without changing the channels
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

            # Reduce with DoubleConv using mid_channels = in_channels // 2 to control param count
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)

        # Trilinear is False
        else:
            # Transposed convolutin upsamples and reduces channels by 2
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            # Post upsampling, concat with skip, giving a total of in_channels
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        """
            Upsample x1, add padding to patch x2, concat, and then finally convolve
            Arguments:
                x1 (torch.Tensor): decoder feature map to upsample (N, C1, T1, H1, W1)
                x2 (torch.Tensor): skip connection feature map from encoder (N, C2, T2, H2, W2)
            Returns:
                torch.Tensor: Output tensor of shape roughly
        """
        # Upsample decoder feature by 2x for T, H, W
        x1 = self.up(x1)

        # Compute temporal dimension difference
        diffT = x2.size()[2] - x1.size()[2] 

        # Height difference
        diffY = x2.size()[3] - x1.size()[3]

        # Width difference
        diffX = x2.size()[4] - x1.size()[4]

        # Pad x1 so it will match 2x in T, H, W before concat
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffT // 2, diffT - diffT // 2])
        
        # Concat along channel dimensions (N, C_2 + C_1, T, H, W)
        x = torch.cat([x2, x1], dim=1)

        # Fuse concat features with two conv layers
        return self.conv(x)

class OutConv3D(nn.Module):
    """
        Final 1x1x1 convolution to project features to the target channels
    """
    def __init__(self, in_channels, out_channels):
        """
            Arguments:
                in_channels (int): number of input channels (C_in) ) duh
                out_channels (int): number of output channels (C_out) you get the vibe now
            Attributes:
                conv (nn.Conv3d): 1x1x1 convoolution that preserves T, W, H
        """
        # Init intialization
        super(OutConv3D, self).__init__()

        # 1x1x1 convolution that changes only the channels 
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
            Project to desired output channels while preserving size
            Arguments:
                x (torch.Tensor): input tensor of shape (N, C_in, T, H, W)
            Returns:
                torch.Tensor: output tensor of shape (N, C_out, T, H, W)
        """
        # Linear projection along the channel axis
        return self.conv(x)

class UNet3D(nn.Module):
    """
        3D U-Net for the volumetric data (H2RG data cubes!!)
        The encoder progessively downsamples to capture context, the decoder upsamples and 
        fuses encoder features (skip connections) to recover detail
    """
    def __init__(self, n_channels, n_classes, trilinear=False):
        """
            Arguments:
                n_channels (int): number of input channels
                n_classes (int): mumber of output channels
                trilinear (bool, optiona): if True, uses trilinear interpolation is explained above
            Attributes:
                n_channels (int): stored input channel count
                n_classes (int): stored output channel count
                trilinear (bool): decoder upsampling mode flag
                inc (DoubleConv3D): initial two-conv block (C: n_channels -> 32)
                down1 (Down3D): downsampling block (32 -> 64)
                down2 (Down3D): downsampling block (64 -> 128)
                down3 (Down3D): downsampling block (128 -> 256)
                down4 (Down3D): downsampling block to bottleneck (256 -> 512//factor)
                up1 (Up3D): upsampling block (concats with 256-skip, outputs 256//factor)
                up2 (Up3D): upsampling block (concats with 128-skip, outputs 128//factor)
                up3 (Up3D): upsampling block (concats with 64-skip, outputs 64//factor)
                up4 (Up3D): upsampling block (concats with 32-skip, outputs 32)
                outc (OutConv3D): final 1x1x1 projection to n_classes
            Notes:
                * if trilinear = True, factor = 2 is used to reduce channels at the bottlenecks
                to keep param count comparabale to its transposed sister
        """
        # Initialize miss init
        super(UNet3D, self).__init__()
        
        # Store configutation (1 grayscale, 3 rgb)
        self.n_channels = n_channels

        # Number of segmentation classes
        self.n_classes = n_classes

        # Decoder upsampling mode
        self.trilinear = trilinear

        """Encoder"""
        # First convolution block
        self.inc = DoubleConv3D(n_channels, 32)
        
        # Downsample by 1/2 T, H, W
        self.down1 = Down3D(32, 64)

        # Downsample by 1/4 T, H, W
        self.down2 = Down3D(64, 128)

        # Downsample by 1/8 T, H, W
        self.down3 = Down3D(128, 256)

        # Adjust channel factor if using trilinear upsampling to balance params
        factor = 2 if trilinear else 1

        # Bottleneck at 1/16 T, H, W
        self.down4 = Down3D(256, 512 // factor)

        """Decoder"""
        # Concat with skip 256
        self.up1 = Up3D(512, 256 // factor, trilinear)

        # Concat with 128
        self.up2 = Up3D(256, 128 // factor, trilinear)

        # Concat with 64
        self.up3 = Up3D(128, 64 // factor, trilinear)

        # Concat with 32
        self.up4 = Up3D(64, 32, trilinear)

        # Final projection to class and channels
        self.outc = OutConv3D(32, n_classes)

    def forward(self, x):
        """
            Full encoder-decoder pass with skip connections
            Arguments:
                x (torch.Tensor): input tensor of shape (N, C_in, T, H, W)
            Returns: 
                torch.Tensor: output tensor of shape (N, C_out, T, H, W) duh
        """
        """Encoder: compute and store skip features"""
        # Skip 1: (N, 32,, T, H, W)
        x1 = self.inc(x)

        # Skip 2: (N, 64, T/2, H/2, W/2)
        x2 = self.down1(x1)

        # Skip 3: (N, 128, T/4, H/4, W/4)
        x3 = self.down2(x2)

        # Skip 4: (N, 256, T/16, H/16, W/16)
        x4 = self.down3(x3)

        # Skip 4: (N, 512 // factor, T/16, H/16, W/16)
        x5 = self.down4(x4)

        """Decoder: upsample and fuse w/ the encoder skip features"""
        # Upsample 1: (N, 256 // factor, T/8, H/8, W/8)
        x = self.up1(x5, x4)

        # Upsample 2: (N, 128 // factor, T/4, H/4, W/4)
        x = self.up2(x, x3)

        # Upsample 3: (N, 64 // factor, T/2, H/2, W/2)
        x = self.up3(x, x2)

        # Upsample 4: (N, 32, T, H, W)
        x = self.up4(x, x1)

        # Output are the per-pixel/per-time step prediction
        logits = self.outc(x)

        return logits