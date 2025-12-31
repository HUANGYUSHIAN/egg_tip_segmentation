"""
從 mask 圖像生成 YOLO segmentation labels
根據 /masks 生成 /labels
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def extract_contours_from_mask(mask, class_id):
    """
    從 mask 中提取指定類別的輪廓
    
    Args:
        mask: mask 圖像 (H, W)，值為類別 ID (0, 1, 2, ...)
        class_id: 要提取的類別 ID
    
    Returns:
        list: 輪廓列表，每個輪廓是 numpy array，形狀為 (N, 1, 2)
    """
    # 創建二值圖（只包含該類別）
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    
    # 使用 cv2.findContours 提取輪廓
    # RETR_EXTERNAL: 只提取外層輪廓
    # CHAIN_APPROX_SIMPLE: 簡化輪廓點
    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    return contours


def contour_to_yolo_format(contour, img_width, img_height):
    """
    將 OpenCV 輪廓轉換為 YOLO segmentation 格式（歸一化座標）
    
    Args:
        contour: OpenCV 輪廓，形狀為 (N, 1, 2)
        img_width: 圖像寬度
        img_height: 圖像高度
    
    Returns:
        list: 歸一化的座標列表 [x1, y1, x2, y2, ...]
    """
    # 輪廓點形狀為 (N, 1, 2)，需要轉換為 (N, 2)
    if len(contour.shape) == 3:
        points = contour.reshape(-1, 2)
    else:
        points = contour
    
    # 轉換為歸一化座標
    normalized_coords = []
    for point in points:
        x, y = point[0], point[1]
        # 歸一化到 [0, 1]
        norm_x = x / img_width
        norm_y = y / img_height
        # 確保在 [0, 1] 範圍內
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))
        normalized_coords.extend([norm_x, norm_y])
    
    return normalized_coords


def generate_label_from_mask(mask_path, img_path, num_classes=3, min_area=10):
    """
    從 mask 圖像生成 YOLO label 文件
    
    Args:
        mask_path: mask 圖像路徑
        img_path: 對應的圖像路徑（用於獲取尺寸）
        num_classes: 類別數量（預設 3: background, egg, tip）
        min_area: 最小輪廓面積（像素），小於此值的輪廓會被忽略
    
    Returns:
        list: YOLO 格式的標籤行列表，每個元素為字符串
    """
    # 讀取 mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'無法讀取 mask: {mask_path}')
    
    # 讀取圖像以獲取尺寸（如果 mask 和圖像尺寸不同，使用 mask 的尺寸）
    img = cv2.imread(str(img_path))
    if img is not None:
        img_height, img_width = img.shape[:2]
    else:
        # 如果無法讀取圖像，使用 mask 的尺寸
        img_height, img_width = mask.shape[:2]
    
    # 確保 mask 和圖像尺寸一致
    mask_height, mask_width = mask.shape[:2]
    if mask_width != img_width or mask_height != img_height:
        # 如果尺寸不一致，調整 mask 尺寸
        mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
    
    label_lines = []
    
    # 對每個類別（跳過背景 class 0）
    for class_id in range(1, num_classes):
        # 提取該類別的輪廓
        contours = extract_contours_from_mask(mask, class_id)
        
        # 對每個輪廓生成一行 label
        for contour in contours:
            # 過濾太小的輪廓（可選，避免噪聲）
            if cv2.contourArea(contour) < min_area:
                continue
            
            # 轉換為 YOLO 格式
            normalized_coords = contour_to_yolo_format(contour, img_width, img_height)
            
            # 格式：class_id x1 y1 x2 y2 ...
            if len(normalized_coords) >= 6:  # 至少需要 3 個點（6 個座標值）
                line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords])
                label_lines.append(line)
    
    return label_lines


def generate_labels(data_dir, split='train', num_classes=3, min_area=10):
    """
    為指定數據集分割生成 labels
    
    Args:
        data_dir: 數據根目錄
        split: 數據集分割 ('train', 'val', 'test')
        num_classes: 類別數量
        min_area: 最小輪廓面積（像素）
    """
    data_path = Path(data_dir)
    images_dir = data_path / split / 'images'
    masks_dir = data_path / split / 'masks'
    labels_dir = data_path / split / 'labels'
    
    # 檢查目錄是否存在
    if not images_dir.exists():
        raise ValueError(f'images 目錄不存在: {images_dir}')
    if not masks_dir.exists():
        raise ValueError(f'masks 目錄不存在: {masks_dir}')
    
    # 創建 labels 目錄
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有 mask 文件
    mask_files = sorted(list(masks_dir.glob('*.png')) + list(masks_dir.glob('*.jpg')))
    
    if len(mask_files) == 0:
        print(f'警告: 在 {masks_dir} 中找不到任何 mask 文件')
        return
    
    print(f'處理 {split} 數據集: {len(mask_files)} 張 mask')
    
    success_count = 0
    error_count = 0
    
    for mask_path in tqdm(mask_files, desc=f'生成 {split} labels'):
        # 對應的圖像文件
        img_path = images_dir / mask_path.name
        
        if not img_path.exists():
            print(f'警告: 找不到對應的圖像文件: {img_path}')
            error_count += 1
            continue
        
        try:
            # 生成 label
            label_lines = generate_label_from_mask(mask_path, img_path, num_classes, min_area)
            
            # 保存 label 文件
            label_path = labels_dir / f'{mask_path.stem}.txt'
            with open(label_path, 'w') as f:
                f.write('\n'.join(label_lines))
                if label_lines:  # 如果不是空文件，添加最後的換行
                    f.write('\n')
            
            success_count += 1
            
        except Exception as e:
            print(f'錯誤: 處理 {mask_path.name} 時發生錯誤: {e}')
            error_count += 1
    
    print(f'\n{split} 數據集處理完成:')
    print(f'  成功: {success_count} 張')
    print(f'  失敗: {error_count} 張')


def check_directories(data_dir, splits=['train', 'val', 'test']):
    """
    檢查必要的目錄是否存在
    
    Args:
        data_dir: 數據根目錄
        splits: 要檢查的數據集分割列表
    
    Returns:
        bool: 所有目錄都存在返回 True
    """
    data_path = Path(data_dir)
    all_exist = True
    
    for split in splits:
        images_dir = data_path / split / 'images'
        masks_dir = data_path / split / 'masks'
        
        if not images_dir.exists():
            print(f'錯誤: {images_dir} 不存在')
            all_exist = False
        
        if not masks_dir.exists():
            print(f'錯誤: {masks_dir} 不存在')
            all_exist = False
    
    return all_exist


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='從 mask 圖像生成 YOLO segmentation labels')
    parser.add_argument('--data-dir', type=str, default='Final',
                       help='數據根目錄')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='要處理的數據集分割 (預設: train val test)')
    parser.add_argument('--num-classes', type=int, default=3,
                       help='類別數量 (預設: 3)')
    parser.add_argument('--min-area', type=int, default=10,
                       help='最小輪廓面積（像素），小於此值的輪廓會被忽略 (預設: 10)')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('Mask 圖像轉 YOLO Labels')
    print('=' * 60)
    print(f'數據目錄: {args.data_dir}')
    print(f'處理分割: {args.split}')
    print(f'類別數量: {args.num_classes}')
    print('=' * 60)
    print()
    
    # 檢查目錄
    print('檢查目錄...')
    if not check_directories(args.data_dir, args.split):
        print('\n錯誤: 必要的目錄不存在，請先確保 images 和 masks 目錄存在')
        return
    
    print('所有必要目錄存在 [OK]\n')
    
    # 處理每個數據集分割
    for split in args.split:
        print(f'\n處理 {split} 數據集...')
        generate_labels(args.data_dir, split=split, num_classes=args.num_classes, min_area=args.min_area)
    
    print('\n' + '=' * 60)
    print('所有 labels 生成完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()

