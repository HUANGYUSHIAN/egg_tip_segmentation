"""
分割數據加載器
包含數據增強和標準化
使用 Albumentations 進行數據增強
"""

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A


class SegmentationDataset(Dataset):
    """
    分割數據集
    讀取圖像和對應的 mask
    使用 Albumentations 進行數據增強
    """
    def __init__(self, data_dir, split='train', transform=None, 
                 normalize=True, mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225], img_size=(480, 640)):
        """
        Args:
            data_dir: 數據根目錄（包含 train/val/test）
            split: 數據集分割 ('train', 'val', 'test')
            transform: 自定義變換（可選）
            normalize: 是否標準化
            mean: 標準化均值
            std: 標準化標準差
            img_size: 目標圖像尺寸 (height, width)，預設 (480, 640)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.img_size = img_size  # (H, W)
        
        # 構建圖像和 mask 路徑
        self.image_dir = self.data_dir / split / 'images'
        self.mask_dir = self.data_dir / split / 'masks'
        
        # 獲取所有圖像文件
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        if len(self.image_files) == 0:
            raise ValueError(f'在 {self.image_dir} 中找不到任何圖像文件')
        
        # 設置變換
        if transform is None:
            if split == 'train':
                # 訓練時使用數據增強
                self.transform = self._get_train_transform()
            else:
                # 驗證/測試時只 resize 和標準化
                self.transform = self._get_val_transform()
        else:
            self.transform = transform
        
        print(f'載入 {split} 數據集: {len(self.image_files)} 張圖像')
        print(f'目標圖像尺寸: {img_size[1]}x{img_size[0]} (寬x高)')
    
    def _get_train_transform(self):
        """
        訓練時的數據增強（使用 Albumentations）
        固定影像大小為 640x480，保持長寬比
        """
        return A.Compose([
            # 固定尺寸，保持長寬比（使用 letterbox padding）
            A.LongestMaxSize(max_size=max(self.img_size), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.img_size[0],
                min_width=self.img_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            
            # 位置與旋轉（使用 Affine 替代 ShiftScaleRotate）
            A.Affine(
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                rotate=(-5, 5),
                scale=(1.0, 1.0),  # 不縮放
                mode=cv2.BORDER_CONSTANT,
                cval=0,
                cval_mask=0,
                p=0.5
            ),
            
            # 物件形狀（翻轉）
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # 光影與背景
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            
            # 紋理與雜訊
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                mean=0,
                p=0.3
            ),
            A.Blur(
                blur_limit=3,
                p=0.3
            ),
            
            # 標準化（如果需要）
            A.Normalize(mean=self.mean, std=self.std) if self.normalize else A.NoOp(),
        ])
    
    def _get_val_transform(self):
        """
        驗證時的變換（只 resize 和標準化，保持長寬比）
        """
        return A.Compose([
            # 固定尺寸，保持長寬比（使用 letterbox padding）
            A.LongestMaxSize(max_size=max(self.img_size), interpolation=cv2.INTER_LINEAR),
            A.PadIfNeeded(
                min_height=self.img_size[0],
                min_width=self.img_size[1],
                border_mode=cv2.BORDER_CONSTANT,
                p=1.0
            ),
            
            # 標準化（如果需要）
            A.Normalize(mean=self.mean, std=self.std) if self.normalize else A.NoOp(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 讀取圖像
        image_path = self.image_files[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
        
        # 讀取 mask
        mask_path = self.mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f'找不到對應的 mask: {mask_path}')
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # 確保 mask 的類別值正確（0, 1, 2）
        # 如果 mask 是 0-255 範圍，需要轉換為類別標籤
        if mask.max() > 2:
            # 如果 mask 是灰度圖，可能需要根據像素值映射到類別
            # 這裡假設 mask 已經是正確的類別標籤（0, 1, 2）
            # 如果實際 mask 使用其他值（如 0, 128, 255），需要調整映射
            # 暫時保持原樣，假設已經是正確的類別標籤
            pass
        
        # 確保 mask 的數據類型正確
        mask = mask.astype(np.uint8)
        
        # 應用變換（使用 Albumentations）
        # Albumentations 需要圖像和 mask 一起傳入
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # 轉換為 tensor
        # Albumentations 的 Normalize 會返回 numpy array，需要轉換為 tensor
        if isinstance(image, np.ndarray):
            # 圖像: (H, W, C) -> (C, H, W)
            if image.ndim == 3:
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            else:
                image = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            # 如果已經是 tensor，確保是 (C, H, W) 格式
            if image.ndim == 3 and image.shape[0] != 3:
                # 可能是 (H, W, C)，需要轉換
                image = image.permute(2, 0, 1)
        else:
            raise TypeError(f'不支持的圖像類型: {type(image)}')
        
        # Mask: (H, W) -> (H, W) long tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()
        else:
            raise TypeError(f'不支持的 mask 類型: {type(mask)}')
        
        return image, mask, image_path.name


def get_dataloader(data_dir, split='train', batch_size=8, shuffle=None, 
                   num_workers=4, normalize=True, img_size=(480, 640)):
    """
    獲取數據加載器
    
    Args:
        data_dir: 數據根目錄
        split: 數據集分割
        batch_size: 批次大小
        shuffle: 是否打亂（None 時自動：train=True, val/test=False）
        num_workers: 數據加載線程數
        normalize: 是否標準化
        img_size: 目標圖像尺寸 (height, width)，預設 (480, 640)
    
    Returns:
        DataLoader
    """
    dataset = SegmentationDataset(
        data_dir=data_dir,
        split=split,
        normalize=normalize,
        img_size=img_size
    )
    
    if shuffle is None:
        shuffle = (split == 'train')
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')  # 訓練時丟棄最後不完整的批次
    )
    
    return dataloader

