"""
DeepLabV3 訓練腳本
包含學習率調度器和自定義 loss

使用方法：
1. 直接在文件中設定參數（推薦），然後運行：python DeepLabV3_train.py
2. 或在 Spyder 中直接執行
3. 或使用命令行參數覆蓋文件中的設定
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# 嘗試導入 tensorboard（可選）
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print('警告: tensorboard 未安裝，將跳過 TensorBoard 日誌記錄')

from DeepLabV3_model import DeepLabV3
from seg_loader import get_dataloader

# ============================================================================
# 訓練參數設定（可在這裡修改）
# ============================================================================
# 數據相關
DATA_DIR = 'Final'  # 數據根目錄

# 模型相關
BACKBONE = 'resnet50'  # 'resnet50' 或 'resnet101'
NUM_CLASSES = 3  # 類別數（background, egg, tip）

# 訓練參數
EPOCHS = 20  # 訓練輪數
BATCH_SIZE = 8  # 批次大小
LEARNING_RATE = 0.001  # 初始學習率
WEIGHT_DECAY = 1e-4  # 權重衰減

# 學習率調度器
SCHEDULER_TYPE = 'cosine'  # 'cosine', 'step', 'plateau'
SCHEDULER_PARAMS = {}  # 調度器參數（JSON 格式字符串或字典）

# Loss 相關
CLASS_WEIGHTS = None  # 類別權重，例如: [1.0, 100.0, 80.0] 或 None（自動計算）
IGNORE_BACKGROUND = True  # 計算 loss 時是否忽略背景類別

# 設備和保存
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 'cuda' 或 'cpu'
SAVE_DIR = 'checkpoints'  # 模型保存目錄
SAVE_INTERVAL = 5  # 每 N 個 epoch 保存一次模型
RESUME = None  # 恢復訓練的檢查點路徑，例如: 'checkpoints/checkpoint_epoch_50.pth'

# 其他
NUM_WORKERS = 4  # 數據加載線程數
LOG_DIR = 'logs'  # TensorBoard 日誌目錄
USE_TENSORBOARD = True  # 是否使用 TensorBoard（如果可用）
# ============================================================================


class WeightedCrossEntropyLoss(nn.Module):
    """
    加權交叉熵損失
    用於處理類別不平衡問題
    """
    def __init__(self, class_weights=None, ignore_background=False):
        """
        Args:
            class_weights: 類別權重 tensor，形狀為 (num_classes,)
            ignore_background: 是否忽略背景類別（只計算非背景的 loss）
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_background = ignore_background
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        else:
            self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - 模型輸出
            targets: (B, H, W) - 真實標籤
        
        Returns:
            loss: 標量
        """
        # 計算每個像素的 loss
        loss_per_pixel = self.ce_loss(predictions, targets)  # (B, H, W)
        
        if self.ignore_background:
            # 只計算非背景像素的 loss
            non_background_mask = (targets != 0)  # 背景類別是 0
            if non_background_mask.sum() > 0:
                loss = loss_per_pixel[non_background_mask].mean()
            else:
                # 如果沒有非背景像素，返回 0
                loss = torch.tensor(0.0, device=predictions.device, requires_grad=True)
        else:
            loss = loss_per_pixel.mean()
        
        return loss


def calculate_miou(predictions, targets, num_classes=3, ignore_background=True):
    """
    計算 mIOU（平均交並比）
    
    Args:
        predictions: (B, H, W) - 預測類別
        targets: (B, H, W) - 真實標籤
        num_classes: 類別數
        ignore_background: 是否忽略背景類別（只計算 class 1 和 class 2）
    
    Returns:
        miou: 標量
        per_class_iou: 每個類別的 IOU
    """
    ious = []
    per_class_iou = {}
    
    # 計算每個類別的 IOU
    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue  # 跳過背景
        
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union > 0:
            iou = intersection / union
            ious.append(iou.item())
            per_class_iou[cls] = iou.item()
        else:
            per_class_iou[cls] = 0.0
    
    if len(ious) > 0:
        miou = np.mean(ious)
    else:
        miou = 0.0
    
    return miou, per_class_iou


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0.0
    total_miou = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs = model(images)
        
        # 計算 loss
        loss = criterion(outputs, masks)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # 計算 mIOU（只計算非背景）
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            miou, _ = calculate_miou(predictions, masks, num_classes=3, ignore_background=True)
        
        total_loss += loss.item()
        total_miou += miou
        num_batches += 1
        
        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'miou': f'{miou:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    return avg_loss, avg_miou


def validate(model, dataloader, criterion, device):
    """驗證"""
    model.eval()
    total_loss = 0.0
    total_miou = 0.0
    per_class_ious = {1: [], 2: []}  # 只記錄 class 1 和 class 2
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward
            outputs = model(images)
            
            # 計算 loss
            loss = criterion(outputs, masks)
            
            # 計算 mIOU
            predictions = torch.argmax(outputs, dim=1)
            miou, per_class_iou = calculate_miou(predictions, masks, num_classes=3, ignore_background=True)
            
            # 記錄每個類別的 IOU
            for cls in [1, 2]:
                if cls in per_class_iou:
                    per_class_ious[cls].append(per_class_iou[cls])
            
            total_loss += loss.item()
            total_miou += miou
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'miou': f'{miou:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_miou = total_miou / num_batches
    
    # 計算每個類別的平均 IOU
    avg_per_class_iou = {}
    for cls in per_class_ious:
        if len(per_class_ious[cls]) > 0:
            avg_per_class_iou[cls] = np.mean(per_class_ious[cls])
        else:
            avg_per_class_iou[cls] = 0.0
    
    return avg_loss, avg_miou, avg_per_class_iou


def parse_args():
    """
    解析命令行參數
    如果提供命令行參數，會覆蓋文件頂部的設定
    """
    parser = argparse.ArgumentParser(description='DeepLabV3 訓練')
    
    # 數據相關
    parser.add_argument('--data-dir', type=str, default=None,
                       help='數據根目錄（覆蓋文件設定）')
    
    # 模型相關
    parser.add_argument('--backbone', type=str, default=None,
                       choices=['resnet50', 'resnet101'],
                       help='骨幹網絡（覆蓋文件設定）')
    parser.add_argument('--num-classes', type=int, default=None,
                       help='類別數（覆蓋文件設定）')
    
    # 訓練參數
    parser.add_argument('--epochs', type=int, default=None,
                       help='訓練輪數（覆蓋文件設定）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（覆蓋文件設定）')
    parser.add_argument('--lr', type=float, default=None,
                       help='初始學習率（覆蓋文件設定）')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='權重衰減（覆蓋文件設定）')
    
    # 學習率調度器
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['cosine', 'step', 'plateau'],
                       help='學習率調度器類型（覆蓋文件設定）')
    parser.add_argument('--scheduler-params', type=str, default=None,
                       help='調度器參數（JSON 格式，覆蓋文件設定）')
    
    # Loss 相關
    parser.add_argument('--class-weights', type=str, default=None,
                       help='類別權重（JSON 格式，例如: "[1.0, 100.0, 80.0]"）')
    parser.add_argument('--ignore-background', action='store_true', default=None,
                       help='計算 loss 時忽略背景類別（覆蓋文件設定）')
    parser.add_argument('--no-ignore-background', dest='ignore_background', action='store_false',
                       help='計算 loss 時不忽略背景類別')
    
    # 設備和保存
    parser.add_argument('--device', type=str, default=None,
                       help='訓練設備 (cuda/cpu，覆蓋文件設定)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='模型保存目錄（覆蓋文件設定）')
    parser.add_argument('--save-interval', type=int, default=None,
                       help='每 N 個 epoch 保存一次模型（覆蓋文件設定）')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢復訓練的檢查點路徑（覆蓋文件設定）')
    
    # 其他
    parser.add_argument('--num-workers', type=int, default=None,
                       help='數據加載線程數（覆蓋文件設定）')
    parser.add_argument('--log-dir', type=str, default=None,
                       help='TensorBoard 日誌目錄（覆蓋文件設定）')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='不使用 TensorBoard')
    
    return parser.parse_args()


def get_config(args):
    """
    合併文件設定和命令行參數
    命令行參數優先級更高
    """
    config = {
        'data_dir': args.data_dir if args.data_dir is not None else DATA_DIR,
        'backbone': args.backbone if args.backbone is not None else BACKBONE,
        'num_classes': args.num_classes if args.num_classes is not None else NUM_CLASSES,
        'epochs': args.epochs if args.epochs is not None else EPOCHS,
        'batch_size': args.batch_size if args.batch_size is not None else BATCH_SIZE,
        'lr': args.lr if args.lr is not None else LEARNING_RATE,
        'weight_decay': args.weight_decay if args.weight_decay is not None else WEIGHT_DECAY,
        'scheduler': args.scheduler if args.scheduler is not None else SCHEDULER_TYPE,
        'scheduler_params': args.scheduler_params if args.scheduler_params is not None else SCHEDULER_PARAMS,
        'class_weights': args.class_weights,
        'ignore_background': args.ignore_background if args.ignore_background is not None else IGNORE_BACKGROUND,
        'device': args.device if args.device is not None else DEVICE,
        'save_dir': args.save_dir if args.save_dir is not None else SAVE_DIR,
        'save_interval': args.save_interval if args.save_interval is not None else SAVE_INTERVAL,
        'resume': args.resume if args.resume is not None else RESUME,
        'num_workers': args.num_workers if args.num_workers is not None else NUM_WORKERS,
        'log_dir': args.log_dir if args.log_dir is not None else LOG_DIR,
        'use_tensorboard': not args.no_tensorboard and USE_TENSORBOARD and TENSORBOARD_AVAILABLE,
    }
    
    # 驗證必要參數
    if config['data_dir'] is None:
        raise ValueError('data_dir 不能為 None，請在文件頂部設定 DATA_DIR 或使用 --data-dir 參數')
    
    return config


def main():
    args = parse_args()
    config = get_config(args)
    
    # 設置設備
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'使用設備: {device}')
    
    # 創建保存目錄
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 設置 TensorBoard
    writer = None
    if config['use_tensorboard']:
        writer = SummaryWriter(log_dir / datetime.now().strftime('%Y%m%d_%H%M%S'))
        print('TensorBoard 日誌記錄已啟用')
    else:
        print('TensorBoard 日誌記錄已禁用')
    
    # 創建數據加載器
    print('載入數據...')
    train_loader = get_dataloader(
        data_dir=config['data_dir'],
        split='train',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    val_loader = get_dataloader(
        data_dir=config['data_dir'],
        split='val',
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # 創建模型
    print('創建模型...')
    model = DeepLabV3(
        num_classes=config['num_classes'],
        backbone=config['backbone'],
        pretrained=True
    )
    model = model.to(device)
    
    # 設置類別權重
    class_weights = None
    if config['class_weights']:
        if isinstance(config['class_weights'], str):
            class_weights = json.loads(config['class_weights'])
        else:
            class_weights = config['class_weights']
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f'使用類別權重: {class_weights.cpu().numpy()}')
    else:
        # 根據統計數據計算權重（background: 99.67%, egg: 0.15%, tip: 0.19%）
        # 使用反頻率權重
        frequencies = [0.9967, 0.0015, 0.0019]
        class_weights = torch.tensor([1.0 / f for f in frequencies], dtype=torch.float32).to(device)
        # 正規化（最小權重設為 1.0）
        class_weights = class_weights / class_weights.min()
        print(f'自動計算類別權重: {class_weights.cpu().numpy()}')
    
    # 創建損失函數
    criterion = WeightedCrossEntropyLoss(
        class_weights=class_weights,
        ignore_background=config['ignore_background']
    )
    
    # 創建優化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # 創建學習率調度器
    if isinstance(config['scheduler_params'], str):
        scheduler_params = json.loads(config['scheduler_params']) if config['scheduler_params'] else {}
    else:
        scheduler_params = config['scheduler_params'] if config['scheduler_params'] else {}
    
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], **scheduler_params)
    elif config['scheduler'] == 'step':
        step_size = scheduler_params.get('step_size', 30)
        gamma = scheduler_params.get('gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif config['scheduler'] == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', **scheduler_params)
    else:
        scheduler = None
    
    # 恢復訓練
    start_epoch = 0
    best_val_miou = 0.0
    
    if config['resume']:
        print(f'恢復訓練從: {config["resume"]}')
        checkpoint = torch.load(config['resume'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_miou = checkpoint.get('best_val_miou', 0.0)
        print(f'從 epoch {start_epoch} 繼續訓練，最佳 val mIOU: {best_val_miou:.4f}')
    
    # 訓練循環
    print('開始訓練...')
    for epoch in range(start_epoch, config['epochs']):
        # 訓練
        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # 驗證
        val_loss, val_miou, val_per_class_iou = validate(model, val_loader, criterion, device)
        
        # 更新學習率
        if scheduler:
            if config['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 記錄到 TensorBoard
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('mIOU/Train', train_miou, epoch)
            writer.add_scalar('mIOU/Val', val_miou, epoch)
            writer.add_scalar('mIOU/Val_Class1', val_per_class_iou.get(1, 0.0), epoch)
            writer.add_scalar('mIOU/Val_Class2', val_per_class_iou.get(2, 0.0), epoch)
            writer.add_scalar('LearningRate', current_lr, epoch)
        
        # 打印結果
        print(f'\nEpoch {epoch+1}/{config["epochs"]}:')
        print(f'  Train Loss: {train_loss:.4f}, Train mIOU: {train_miou:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val mIOU: {val_miou:.4f}')
        print(f'  Val Class 1 (egg) IOU: {val_per_class_iou.get(1, 0.0):.4f}')
        print(f'  Val Class 2 (tip) IOU: {val_per_class_iou.get(2, 0.0):.4f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # 保存最佳模型
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_model_path = save_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_miou': best_val_miou,
                'val_loss': val_loss,
                'val_miou': val_miou,
            }, best_model_path)
            print(f'  保存最佳模型: {best_model_path}')
        
        # 每 N 個 epoch 保存檢查點
        if (epoch + 1) % config['save_interval'] == 0:
            checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_miou': best_val_miou,
                'val_loss': val_loss,
                'val_miou': val_miou,
            }, checkpoint_path)
            print(f'  保存檢查點: {checkpoint_path}')
    
    # 保存最終模型
    final_model_path = save_dir / 'final_model.pth'
    torch.save({
        'epoch': config['epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_miou': best_val_miou,
    }, final_model_path)
    print(f'\n訓練完成！最終模型保存至: {final_model_path}')
    print(f'最佳驗證 mIOU: {best_val_miou:.4f}')
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    main()

