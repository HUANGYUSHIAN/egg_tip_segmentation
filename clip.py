"""
圖像裁剪工具
從輸入文件夾讀取圖像和 masks，裁剪指定高度範圍後保存到輸出文件夾
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import shutil


def clip_image(image, hclip_ratio):
    """
    裁剪圖像的垂直方向
    
    Args:
        image: 圖像 numpy array，形狀為 (H, W, C) 或 (H, W)
        hclip_ratio: 裁剪比例 (start_ratio, end_ratio)，例如 (0, 0.2) 表示裁剪掉頂部 0% 到 20%
    
    Returns:
        numpy.ndarray: 裁剪後的圖像
    """
    height = image.shape[0]
    start_pixel = int(height * hclip_ratio[0])
    end_pixel = int(height * hclip_ratio[1])
    
    # 裁剪：保留從 end_pixel 到底部的部分（去除 start_pixel 到 end_pixel 的部分）
    if len(image.shape) == 3:
        # 彩色圖像 (H, W, C)
        clipped = np.concatenate([
            image[:start_pixel],
            image[end_pixel:]
        ], axis=0)
    else:
        # 灰度圖像 (H, W)
        clipped = np.concatenate([
            image[:start_pixel],
            image[end_pixel:]
        ], axis=0)
    
    return clipped


def process_split(input_dir, output_dir, split, hclip_ratio):
    """
    處理一個數據集分割（train/val/test）
    
    Args:
        input_dir: 輸入根目錄
        output_dir: 輸出根目錄
        split: 數據集分割名稱 ('train', 'val', 'test')
        hclip_ratio: 裁剪比例 (start_ratio, end_ratio)
    """
    input_images_dir = input_dir / split / 'images'
    input_masks_dir = input_dir / split / 'masks'
    output_images_dir = output_dir / split / 'images'
    output_masks_dir = output_dir / split / 'masks'
    
    # 創建輸出目錄
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取所有圖像文件
    image_files = sorted(
        list(input_images_dir.glob('*.png')) + 
        list(input_images_dir.glob('*.jpg')) +
        list(input_images_dir.glob('*.jpeg'))
    )
    
    if len(image_files) == 0:
        print(f'警告: 在 {input_images_dir} 中找不到任何圖像文件')
        return 0, 0
    
    print(f'\n處理 {split} 數據集: {len(image_files)} 張圖像')
    
    success_count = 0
    error_count = 0
    
    for img_path in tqdm(image_files, desc=f'裁剪 {split} 圖像'):
        try:
            # 讀取圖像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f'警告: 無法讀取圖像 {img_path}')
                error_count += 1
                continue
            
            # 裁剪圖像
            clipped_image = clip_image(image, hclip_ratio)
            
            # 保存裁剪後的圖像
            output_img_path = output_images_dir / img_path.name
            cv2.imwrite(str(output_img_path), clipped_image)
            
            # 處理對應的 mask（如果存在）
            mask_path = input_masks_dir / img_path.name
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # 確保 mask 和圖像尺寸一致
                    if mask.shape[:2] != image.shape[:2]:
                        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                    # 裁剪 mask
                    clipped_mask = clip_image(mask, hclip_ratio)
                    
                    # 保存裁剪後的 mask
                    output_mask_path = output_masks_dir / img_path.name
                    cv2.imwrite(str(output_mask_path), clipped_mask)
            
            success_count += 1
            
        except Exception as e:
            print(f'錯誤: 處理 {img_path.name} 時發生錯誤: {e}')
            error_count += 1
    
    return success_count, error_count


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='裁剪圖像和 masks')
    parser.add_argument('--input', type=str, required=True,
                       help='輸入文件夾路徑（例如: C:\\Users\\huang\\Desktop\\egg detection\\Final）')
    parser.add_argument('--output', type=str, required=True,
                       help='輸出文件夾路徑（例如: C:\\Users\\huang\\Desktop\\egg detection\\Final_clip）')
    parser.add_argument('--hclip-ratio', type=float, nargs=2, default=[0.0, 0.2],
                       metavar=('START', 'END'),
                       help='垂直裁剪比例，例如: 0.0 0.2 表示裁剪掉頂部 0%% 到 20%% 的部分（預設: 0.0 0.2）')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'val', 'test'],
                       help='要處理的數據集分割（預設: train val test）')
    
    args = parser.parse_args()
    
    # 解析輸入和輸出路徑
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    # 驗證輸入目錄
    if not input_dir.exists():
        raise ValueError(f'輸入目錄不存在: {input_dir}')
    
    # 解析裁剪比例
    hclip_ratio = tuple(args.hclip_ratio)
    if hclip_ratio[0] < 0 or hclip_ratio[1] < 0 or hclip_ratio[0] >= hclip_ratio[1] or hclip_ratio[1] > 1.0:
        raise ValueError(f'無效的裁剪比例: {hclip_ratio}，必須滿足 0 <= start < end <= 1.0')
    
    print('=' * 60)
    print('圖像裁剪工具')
    print('=' * 60)
    print(f'輸入目錄: {input_dir}')
    print(f'輸出目錄: {output_dir}')
    print(f'裁剪比例: {hclip_ratio[0]:.1%} ~ {hclip_ratio[1]:.1%} (將裁剪掉此範圍)')
    print(f'處理分割: {args.split}')
    print('=' * 60)
    
    # 創建輸出目錄
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'\n輸出目錄已創建: {output_dir}')
    
    # 處理每個數據集分割
    total_success = 0
    total_error = 0
    
    for split in args.split:
        input_split_dir = input_dir / split
        if not input_split_dir.exists():
            print(f'\n警告: 跳過不存在的分割: {split}')
            continue
        
        success, error = process_split(input_dir, output_dir, split, hclip_ratio)
        total_success += success
        total_error += error
        
        if success > 0:
            print(f'  {split}: 成功 {success} 張，失敗 {error} 張')
    
    print('\n' + '=' * 60)
    print('裁剪完成!')
    print('=' * 60)
    print(f'總計: 成功 {total_success} 張，失敗 {total_error} 張')
    print(f'輸出目錄: {output_dir}')


if __name__ == '__main__':
    main()

