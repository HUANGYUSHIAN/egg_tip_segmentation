"""
DeepLabV3 模型定義
基於 torchvision 的 DeepLabV3，用於 3 類別分割任務（background, egg, tip）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.segmentation as segmentation


class DeepLabV3(nn.Module):
    """
    DeepLabV3 模型
    輸入: RGB 圖像 (3 channels)
    輸出: 3 類別分割結果 (background, egg, tip)
    """
    def __init__(self, num_classes=3, backbone='resnet50', pretrained=True):
        """
        Args:
            num_classes: 輸出類別數（預設 3: background, egg, tip）
            backbone: 骨幹網絡 ('resnet50' 或 'resnet101')
            pretrained: 是否使用預訓練權重
        """
        super(DeepLabV3, self).__init__()
        
        # 載入預訓練的 DeepLabV3
        if backbone == 'resnet50':
            self.deeplab = segmentation.deeplabv3_resnet50(
                weights='DEFAULT' if pretrained else None,
                num_classes=21  # 預設是 21 classes (COCO)
            )
        elif backbone == 'resnet101':
            self.deeplab = segmentation.deeplabv3_resnet101(
                weights='DEFAULT' if pretrained else None,
                num_classes=21
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet50' or 'resnet101'")
        
        # 修改分類器輸出層以匹配我們的類別數
        # DeepLabV3 的 classifier 是最後一層 Conv2d
        original_conv = self.deeplab.classifier[4]
        self.deeplab.classifier[4] = nn.Conv2d(
            original_conv.in_channels,
            num_classes,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # 如果有 aux_classifier，也修改它
        if hasattr(self.deeplab, 'aux_classifier') and self.deeplab.aux_classifier is not None:
            aux_original_conv = self.deeplab.aux_classifier[4]
            self.deeplab.aux_classifier[4] = nn.Conv2d(
                aux_original_conv.in_channels,
                num_classes,
                kernel_size=aux_original_conv.kernel_size,
                stride=aux_original_conv.stride,
                padding=aux_original_conv.padding,
                bias=aux_original_conv.bias is not None
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 輸入 tensor (B, 3, H, W) - RGB 圖像
        
        Returns:
            output: 分割結果 (B, num_classes, H, W)
        """
        # DeepLabV3 的 forward 返回 OrderedDict，包含 'out' 和可能的 'aux'
        output = self.deeplab(x)
        
        # 提取主要輸出
        if isinstance(output, dict):
            output = output['out']
        
        # 確保輸出尺寸與輸入一致（DeepLabV3 輸出可能比輸入小）
        if output.shape[2:] != x.shape[2:]:
            output = F.interpolate(
                output,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        return output

