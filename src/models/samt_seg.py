import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# CBAM: Convolutional Block Attention Module
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

# ==============================================================================
# UNet Decoder Block
# ==============================================================================
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels // 2 + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle padding if dimensions don't perfectly match
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.cbam(x)
        return x

# ==============================================================================
# SAMTSeg: Severity-Aware Multi-Task Segmentation Network
# ==============================================================================
class SAMTSeg(nn.Module):
    """
    Novel Architecture: Severity-Aware Multi-Task Segmentation Network
    Features:
    - Pre-trained backbone (here simulated with simple CNN blocks for standalone execution, 
      in practice use torchvision.models.efficientnet_b4)
    - CBAM attention mechanisms
    - Segmentation Decoder
    - Severity Regression Head
    """
    def __init__(self, num_classes=10):
        super(SAMTSeg, self).__init__()
        
        # NOTE: In a full implementation, replace this simple encoder with 
        # a pre-trained EfficientNet or ResNet backbone using `timm` or `torchvision`.
        # This is a simplified ResNet-like encoder for structural demonstration.
        
        self.enc1 = self._conv_block(3, 64)      # H, W
        self.pool1 = nn.MaxPool2d(2, 2)          # H/2, W/2
        
        self.enc2 = self._conv_block(64, 128)    # H/2, W/2
        self.pool2 = nn.MaxPool2d(2, 2)          # H/4, W/4
        
        self.enc3 = self._conv_block(128, 256)   # H/4, W/4
        self.pool3 = nn.MaxPool2d(2, 2)          # H/8, W/8
        
        self.enc4 = self._conv_block(256, 512)   # H/8, W/8
        self.pool4 = nn.MaxPool2d(2, 2)          # H/16, W/16
        
        self.bottleneck = self._conv_block(512, 1024) # H/16, W/16
        self.bottleneck_cbam = CBAM(1024)
        
        # --- Task 1: Segmentation Decoder ---
        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        self.seg_head = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # --- Task 2: Severity Regression Head ---
        # Takes the bottleneck features (highest level of abstraction)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.sev_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Outputs severity ratio [0, 1]
        )

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool4(e4))
        b = self.bottleneck_cbam(b)
        
        # Task 1: Segmentation
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        seg_mask = self.seg_head(d1)  # Logits, shape: (B, num_classes, H, W)
        
        # Task 2: Severity Grading
        # Extract features from the bottleneck
        sev_feat = self.global_pool(b).view(b.size(0), -1)
        sev_score = self.sev_head(sev_feat)  # Shape: (B, 1)
        
        return seg_mask, sev_score


# ==============================================================================
# Severity-Consistency Loss
# ==============================================================================
class SeverityConsistencyLoss(nn.Module):
    def __init__(self, background_class_idx=0):
        super(SeverityConsistencyLoss, self).__init__()
        self.bg_idx = background_class_idx
        self.mse = nn.MSELoss()

    def forward(self, predicted_logits, predicted_severity):
        """
        predicted_logits: (B, C, H, W) segmentation logits
        predicted_severity: (B, 1) scalar severity score [0,1]
        """
        B, C, H, W = predicted_logits.size()
        
        # Convert logits to probabilities
        probs = F.softmax(predicted_logits, dim=1)
        
        # Sum probabilities of all disease classes (everything except background)
        # This gives a differentiable approximation of "disease pixels"
        disease_probs = 1.0 - probs[:, self.bg_idx, :, :]  # Shape: (B, H, W)
        
        # Compute inferred severity ratio
        # Sum over spatial dimensions and divide by area
        inferred_severity = disease_probs.sum(dim=(1, 2)) / (H * W)  # Shape: (B,)
        inferred_severity = inferred_severity.view(-1, 1)  # Shape: (B, 1)
        
        # MSE between the regression head output and the segmentation-inferred severity
        return self.mse(predicted_severity, inferred_severity)


if __name__ == "__main__":
    # Test the model
    model = SAMTSeg(num_classes=10)
    x = torch.randn(2, 3, 256, 256)
    seg, sev = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Segmentation Output: {seg.shape}")
    print(f"Severity Output: {sev.shape}")
    
    # Test loss
    loss_fn = SeverityConsistencyLoss()
    cons_loss = loss_fn(seg, sev)
    print(f"Consistency Loss: {cons_loss.item():.4f}")
