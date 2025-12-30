"""
從 YOLO labels 生成 mask 圖像
根據 /images 和 /labels 生成 /masks
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def parse_yolo_label(label_path, img_width, img_height):
    """
    解析 YOLO 格式的 label 文件
    
    Args:
        label_path: label 文件路徑
        img_width: 圖像寬度
        img_height: 圖像高度
    
    Returns:
        list: 每個元素為 (class_id, polygon_points)，polygon_points 是絕對座標
    """
    polygons = []
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 3:
                continue
            
            class_id = int(parts[0])
            # 剩餘的是歸一化的多邊形座標 (x1, y1, x2, y2, ...)
            coords = [float(x) for x in parts[1:]]
            
            # 轉換為絕對座標
            polygon_points = []
            for i in range(0, len(coords), 2):
                if i + 1 < len(coords):
                    x = coords[i] * img_width
                    y = coords[i + 1] * img_height
                    polygon_points.append([int(x), int(y)])
            
            if len(polygon_points) >= 3:  # 至少需要 3 個點才能形成多邊形
                polygons.append((class_id, np.array(polygon_points, dtype=np.int32)))
    
    return polygons


def create_mask_from_labels(label_path, img_path, num_classes=3):
    """
    從 YOLO label 創建 mask 圖像
    
    Args:
        label_path: label 文件路徑
        img_path: 對應的圖像路徑
        num_classes: 類別數量（預設 3: background, egg, tip）
    
    Returns:
        numpy.ndarray: mask 圖像，值為 0, 1, 2 等類別 ID
    """
    # 讀取圖像以獲取尺寸
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f'無法讀取圖像: {img_path}')
    
    img_height, img_width = img.shape[:2]
    
    # 創建空白 mask（背景為 0）
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # 解析 label 文件
    polygons = parse_yolo_label(label_path, img_width, img_height)
    
    # 繪製每個多邊形
    for class_id, polygon_points in polygons:
        # 使用 cv2.fillPoly 填充多邊形
        # class_id 直接作為像素值（0=background, 1=egg, 2=tip）
        cv2.fillPoly(mask, [polygon_points], int(class_id))
    
    return mask


def generate_masks(data_dir, split='train', num_classes=3):
    """
    為指定數據集分割生成 masks
    
    Args:
        data_dir: 數據根目錄
        split: 數據集分割 ('train', 'val', 'test')
        num_classes: 類別數量
    """
    data_path = Path(data_dir)
    images_dir = data_path / split / 'images'
    labels_dir = data_path / split / 'labels'
    masks_dir = data_path / split / 'masks'
    
    # 創建 masks 目錄
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖像文件
    image_files = sorted(list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg')))
    
    if len(image_files) == 0:
        print(f'警告: 在 {images_dir} 中找不到任何圖像文件')
        return
    
    print(f'處理 {split} 數據集: {len(image_files)} 張圖像')
    
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc=f'生成 {split} masks'):
        # 對應的 label 文件
        label_path = labels_dir / f'{img_path.stem}.txt'
        
        if not label_path.exists():
            print(f'警告: 找不到對應的 label 文件: {label_path}')
            error_count += 1
            continue
        
        try:
            # 生成 mask
            mask = create_mask_from_labels(label_path, img_path, num_classes)
            
            # 保存 mask
            mask_path = masks_dir / f'{img_path.stem}.png'
            cv2.imwrite(str(mask_path), mask)
            
            success_count += 1
            
        except Exception as e:
            print(f'錯誤: 處理 {img_path.name} 時發生錯誤: {e}')
            error_count += 1
    
    print(f'\n{split} 數據集處理完成:')
    print(f'  成功: {success_count} 張')
    print(f'  失敗: {error_count} 張')


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='從 YOLO labels 生成 mask 圖像')
    parser.add_argument('--data-dir', type=str, default='Final',
                       help='數據根目錄')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='要處理的數據集分割 (預設: train val test)')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='類別數量 (預設: 3)')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('YOLO Labels 轉 Mask 圖像')
    print('=' * 60)
    print(f'數據目錄: {args.data_dir}')
    print(f'處理分割: {args.split}')
    print(f'類別數量: {args.num_classes}')
    print('=' * 60)
    print()
    
    # 處理每個數據集分割
    for split in args.split:
        print(f'\n處理 {split} 數據集...')
        generate_masks(args.data_dir, split=split, num_classes=args.num_classes)
    
    print('\n' + '=' * 60)
    print('所有 masks 生成完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()

