"""
YOLOv11 圖像分割訓練腳本
使用 ultralytics 進行 YOLO segmentation 訓練
影像固定尺寸: 640x480
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

# 固定圖像尺寸為 640x480
IMG_WIDTH = 640
IMG_HEIGHT = 480


def parse_args():
    """解析命令行參數 - 僅保留最重要的 5 個參數"""
    parser = argparse.ArgumentParser(description='YOLOv11 圖像分割訓練')
    
    # 最重要的 5 個參數
    parser.add_argument('--epochs', type=int, default=10,
                       help='訓練輪數 (epochs)')
    parser.add_argument('--batch', type=int, default=16,
                       help='批次大小 (batch size)')
    parser.add_argument('--model', type=str, default='yolo11n-seg.pt',
                       help='模型文件 (yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='訓練設備 (0, 1, 2, ... 或 cpu)')
    parser.add_argument('--data', type=str, default='dataset.yaml',
                       help='數據集配置文件路徑')
    
    return parser.parse_args()


def main():
    """主訓練函數"""
    args = parse_args()
    
    # 檢查數據集配置文件
    data_path = Path(args.data)
    if not data_path.exists():
        raise ValueError(f'數據集配置文件不存在: {data_path}')
    
    print('=' * 60)
    print('YOLOv11 圖像分割訓練')
    print('=' * 60)
    print(f'模型: {args.model}')
    print(f'訓練輪數: {args.epochs}')
    print(f'批次大小: {args.batch}')
    print(f'圖像尺寸: {IMG_WIDTH}x{IMG_HEIGHT} (固定)')
    print(f'設備: {args.device}')
    print(f'數據集配置: {args.data}')
    print('=' * 60)
    
    # 加載模型
    print(f'\n正在加載模型: {args.model}')
    model = YOLO(args.model)
    
    # 訓練參數 - 固定圖像尺寸為 640x480
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': [IMG_WIDTH, IMG_HEIGHT],  # 固定為 640x480
        'device': args.device,
        'project': 'runs/segment',
        'name': 'train',
        'pretrained': True,
        'amp': True,  # 自動混合精度訓練
    }
    
    # 開始訓練
    print('\n開始訓練...\n')
    results = model.train(**train_kwargs)
    
    print('\n' + '=' * 60)
    print('訓練完成!')
    print('=' * 60)
    print(f'結果保存在: runs/segment/train')
    
    # 驗證最佳模型
    print('\n正在驗證最佳模型...')
    best_model_path = Path('runs/segment/train/weights/best.pt')
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val(data=args.data)
        print(f'\n最佳模型驗證結果:')
        print(f'  mAP50: {metrics.seg.map50:.4f}')
        print(f'  mAP50-95: {metrics.seg.map:.4f}')
    else:
        print('警告: 未找到最佳模型文件')


if __name__ == '__main__':
    main()

