"""
類別像素統計腳本
用於分析分割標籤圖像中每個類別的像素分佈
輸出統計信息以便設計 DeepLabV3 的 loss 權重

使用方法：
1. 命令行模式：
   python class_statistic.py --folder Final/train --class-names background egg broken_egg

2. Spyder 模式（在 Spyder 中直接運行）：
   在文件末尾找到 Spyder 模式區塊，取消註釋並設定路徑即可
   或直接調用：
   result = run_analysis(folder='Final/train', 
                        output='class_statistics.json',
                        class_names=['background', 'egg', 'broken_egg'])
"""

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
import cv2
from tqdm import tqdm
import json


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='統計分割標籤圖像中的類別像素分佈')
    parser.add_argument('--folder', type=str, required=True,
                       help='包含 /masks 資料夾的根目錄路徑')
    parser.add_argument('--output', type=str, default='class_statistics.json',
                       help='輸出統計結果的 JSON 文件路徑')
    parser.add_argument('--class-names', type=str, nargs='+', 
                       default=['background', 'class_0', 'class_1'],
                       help='類別名稱列表 (預設: background class_0 class_1)')
    return parser.parse_args()


def count_objects(mask, class_id):
    """
    計算指定類別的物件數量（連通區域）
    
    Args:
        mask: 二值化 mask（只包含該類別）
        class_id: 類別 ID
        
    Returns:
        int: 物件數量
    """
    # 創建只包含該類別的二值圖
    binary_mask = (mask == class_id).astype(np.uint8) * 255
    
    # 使用連通組件分析
    num_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)
    
    # num_labels 包含背景（標籤0），所以物件數是 num_labels - 1
    # 但我們需要排除太小的連通區域（可能是噪聲）
    # 這裡我們計算所有非零標籤的數量
    if num_labels <= 1:
        return 0
    
    # 計算每個連通區域的大小，過濾掉太小的區域（可選）
    object_count = 0
    for label_id in range(1, num_labels):  # 跳過背景標籤 0
        region_size = np.sum(labels == label_id)
        # 如果區域太小（少於 10 個像素），可能是噪聲，可以選擇過濾
        # 這裡我們不過濾，計算所有連通區域
        if region_size > 0:
            object_count += 1
    
    return object_count


def analyze_mask_image(mask_path):
    """
    分析單張 mask 圖像
    
    Args:
        mask_path: mask 圖像路徑
        
    Returns:
        dict: 包含每個類別的像素數量、物件數量和總像素數
    """
    # 讀取圖像（灰度圖）
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f'警告: 無法讀取圖像 {mask_path}')
        return None
    
    # 獲取唯一的像素值（類別）
    unique_values, counts = np.unique(mask, return_counts=True)
    
    # 轉換為字典
    class_counts = {}  # 像素數量
    class_object_counts = {}  # 物件數量
    total_pixels = mask.size
    
    for value, count in zip(unique_values, counts):
        class_id = int(value)
        class_counts[class_id] = int(count)
        
        # 計算該類別的物件數量（跳過背景類別 0）
        if class_id != 0:  # 背景不需要計算物件數
            object_count = count_objects(mask, class_id)
            class_object_counts[class_id] = object_count
        else:
            class_object_counts[class_id] = 0  # 背景設為 0
    
    return {
        'class_counts': class_counts,  # 像素數量
        'class_object_counts': class_object_counts,  # 物件數量
        'total_pixels': int(total_pixels),
        'shape': mask.shape,
        'image_name': Path(mask_path).name
    }


def calculate_statistics(all_results, class_names):
    """
    計算所有圖像的統計信息
    
    Args:
        all_results: 所有圖像的分析結果列表
        class_names: 類別名稱列表
        
    Returns:
        dict: 統計結果
    """
    # 初始化統計數據
    class_stats = defaultdict(lambda: {
        'pixel_counts': [],
        'proportions': [],
        'object_counts': [],  # 物件數量列表
        'total_pixels': 0,
        'per_image_details': []  # 每張圖的詳細信息
    })
    
    all_classes = set()
    
    # 收集所有類別的數據
    for result in all_results:
        if result is None:
            continue
            
        total_pixels = result['total_pixels']
        class_counts = result['class_counts']
        class_object_counts = result.get('class_object_counts', {})
        image_name = result.get('image_name', 'unknown')
        
        # 更新每個類別的統計
        for class_id, pixel_count in class_counts.items():
            all_classes.add(class_id)
            class_stats[class_id]['pixel_counts'].append(pixel_count)
            class_stats[class_id]['proportions'].append(pixel_count / total_pixels)
            class_stats[class_id]['total_pixels'] += pixel_count
            
            # 物件數量
            object_count = class_object_counts.get(class_id, 0)
            class_stats[class_id]['object_counts'].append(object_count)
            
            # 記錄每張圖的詳細信息
            class_stats[class_id]['per_image_details'].append({
                'image_name': image_name,
                'pixel_count': int(pixel_count),
                'object_count': int(object_count),
                'proportion': float(pixel_count / total_pixels)
            })
    
    # 計算統計量
    statistics = {}
    
    # 總像素數（所有圖像的總和）
    total_all_pixels = sum(
        sum(result['class_counts'].values()) 
        for result in all_results 
        if result is not None
    )
    
    for class_id in sorted(all_classes):
        stats = class_stats[class_id]
        pixel_counts = np.array(stats['pixel_counts'])
        proportions = np.array(stats['proportions'])
        object_counts = np.array(stats['object_counts'])
        
        # 基本統計（像素）
        mean_pixels = np.mean(pixel_counts)
        std_pixels = np.std(pixel_counts)
        median_pixels = np.median(pixel_counts)
        min_pixels = np.min(pixel_counts)
        max_pixels = np.max(pixel_counts)
        
        # 比例統計
        mean_proportion = np.mean(proportions)
        std_proportion = np.std(proportions)
        median_proportion = np.median(proportions)
        
        # 物件數量統計
        mean_objects = np.mean(object_counts)
        std_objects = np.std(object_counts)
        median_objects = np.median(object_counts)
        min_objects = int(np.min(object_counts))
        max_objects = int(np.max(object_counts))
        total_objects = int(np.sum(object_counts))
        
        # 總體比例
        total_class_pixels = stats['total_pixels']
        overall_proportion = total_class_pixels / total_all_pixels if total_all_pixels > 0 else 0
        
        # 類別名稱
        if class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f'class_{class_id}'
        
        statistics[class_id] = {
            'class_id': int(class_id),
            'class_name': class_name,
            'total_pixels': int(total_class_pixels),
            'overall_proportion': float(overall_proportion),
            'pixel_statistics': {
                'mean': float(mean_pixels),
                'std': float(std_pixels),
                'median': float(median_pixels),
                'min': int(min_pixels),
                'max': int(max_pixels),
            },
            'proportion_statistics': {
                'mean': float(mean_proportion),
                'std': float(std_proportion),
                'median': float(median_proportion),
            },
            'object_statistics': {
                'total': total_objects,
                'mean': float(mean_objects),
                'std': float(std_objects),
                'median': float(median_objects),
                'min': min_objects,
                'max': max_objects,
            },
            'per_image_details': stats['per_image_details'],  # 每張圖的詳細信息
            'num_images': len(pixel_counts)
        }
    
    return statistics, total_all_pixels


def calculate_loss_weights(statistics):
    """
    計算建議的 loss 權重
    
    常用的權重計算方法：
    1. 反頻率權重: weight = 1 / frequency
    2. 平衡權重: weight = median_frequency / frequency
    3. 平方根反頻率: weight = sqrt(1 / frequency)
    
    Args:
        statistics: 統計結果字典
        
    Returns:
        dict: 權重建議
    """
    # 獲取所有類別的總體比例
    proportions = {cid: stats['overall_proportion'] for cid, stats in statistics.items()}
    
    # 方法1: 反頻率權重（最常用）
    inverse_freq_weights = {}
    for cid, prop in proportions.items():
        if prop > 0:
            inverse_freq_weights[cid] = 1.0 / prop
        else:
            inverse_freq_weights[cid] = 0.0
    
    # 方法2: 平衡權重（使用中位數頻率）
    median_freq = np.median(list(proportions.values()))
    balanced_weights = {}
    for cid, prop in proportions.items():
        if prop > 0:
            balanced_weights[cid] = median_freq / prop
        else:
            balanced_weights[cid] = 0.0
    
    # 方法3: 平方根反頻率
    sqrt_inverse_weights = {}
    for cid, prop in proportions.items():
        if prop > 0:
            sqrt_inverse_weights[cid] = np.sqrt(1.0 / prop)
        else:
            sqrt_inverse_weights[cid] = 0.0
    
    # 正規化權重（使最小權重為1）
    def normalize_weights(weights):
        min_weight = min(w for w in weights.values() if w > 0)
        return {cid: w / min_weight for cid, w in weights.items()}
    
    return {
        'inverse_frequency': inverse_freq_weights,
        'balanced': balanced_weights,
        'sqrt_inverse_frequency': sqrt_inverse_weights,
        'normalized_inverse_frequency': normalize_weights(inverse_freq_weights),
        'normalized_balanced': normalize_weights(balanced_weights),
    }


def print_statistics(statistics, total_pixels, loss_weights, class_names):
    """打印統計結果"""
    print('\n' + '=' * 80)
    print('類別像素統計結果')
    print('=' * 80)
    
    print(f'\n總像素數（所有圖像）: {total_pixels:,}')
    print(f'分析的類別數量: {len(statistics)}')
    print('\n' + '-' * 80)
    
    # 打印每個類別的詳細統計
    for class_id in sorted(statistics.keys()):
        stats = statistics[class_id]
        print(f'\n類別 ID: {class_id} - {stats["class_name"]}')
        print(f'  總像素數: {stats["total_pixels"]:,}')
        print(f'  總體比例: {stats["overall_proportion"]:.4%}')
        print(f'  出現圖像數: {stats["num_images"]}')
        print(f'\n  每張圖像的像素數統計:')
        print(f'    平均值: {stats["pixel_statistics"]["mean"]:.2f}')
        print(f'    標準差: {stats["pixel_statistics"]["std"]:.2f}')
        print(f'    中位數: {stats["pixel_statistics"]["median"]:.2f}')
        print(f'    最小值: {stats["pixel_statistics"]["min"]:,}')
        print(f'    最大值: {stats["pixel_statistics"]["max"]:,}')
        print(f'\n  每張圖像的比例統計:')
        print(f'    平均比例: {stats["proportion_statistics"]["mean"]:.4%}')
        print(f'    標準差: {stats["proportion_statistics"]["std"]:.4%}')
        print(f'    中位數比例: {stats["proportion_statistics"]["median"]:.4%}')
        print(f'\n  每張圖像的物件數統計:')
        print(f'    總物件數: {stats["object_statistics"]["total"]:,}')
        print(f'    平均值: {stats["object_statistics"]["mean"]:.2f}')
        print(f'    標準差: {stats["object_statistics"]["std"]:.2f}')
        print(f'    中位數: {stats["object_statistics"]["median"]:.2f}')
        print(f'    最小值: {stats["object_statistics"]["min"]}')
        print(f'    最大值: {stats["object_statistics"]["max"]}')
    
    # 打印權重建議
    print('\n' + '=' * 80)
    print('DeepLabV3 Loss 權重建議')
    print('=' * 80)
    
    print('\n方法 1: 反頻率權重 (Inverse Frequency)')
    print('  公式: weight = 1 / frequency')
    for cid in sorted(loss_weights['inverse_frequency'].keys()):
        name = statistics[cid]['class_name']
        weight = loss_weights['inverse_frequency'][cid]
        print(f'    {name} (ID {cid}): {weight:.4f}')
    
    print('\n方法 2: 平衡權重 (Balanced)')
    print('  公式: weight = median_frequency / frequency')
    for cid in sorted(loss_weights['balanced'].keys()):
        name = statistics[cid]['class_name']
        weight = loss_weights['balanced'][cid]
        print(f'    {name} (ID {cid}): {weight:.4f}')
    
    print('\n方法 3: 平方根反頻率權重 (Square Root Inverse Frequency)')
    print('  公式: weight = sqrt(1 / frequency)')
    for cid in sorted(loss_weights['sqrt_inverse_frequency'].keys()):
        name = statistics[cid]['class_name']
        weight = loss_weights['sqrt_inverse_frequency'][cid]
        print(f'    {name} (ID {cid}): {weight:.4f}')
    
    print('\n方法 4: 正規化反頻率權重 (Normalized Inverse Frequency)')
    print('  最小權重設為 1.0')
    for cid in sorted(loss_weights['normalized_inverse_frequency'].keys()):
        name = statistics[cid]['class_name']
        weight = loss_weights['normalized_inverse_frequency'][cid]
        print(f'    {name} (ID {cid}): {weight:.4f}')
    
    print('\n方法 5: 正規化平衡權重 (Normalized Balanced)')
    print('  最小權重設為 1.0')
    for cid in sorted(loss_weights['normalized_balanced'].keys()):
        name = statistics[cid]['class_name']
        weight = loss_weights['normalized_balanced'][cid]
        print(f'    {name} (ID {cid}): {weight:.4f}')
    
    print('\n' + '=' * 80)
    print('建議: 通常使用「正規化反頻率權重」或「正規化平衡權重」')
    print('=' * 80 + '\n')


def run_analysis(folder, output='class_statistics.json', class_names=None):
    """
    執行統計分析（可在 Spyder 中直接調用）
    
    Args:
        folder: 包含 /masks 資料夾的根目錄路徑
        output: 輸出 JSON 文件路徑
        class_names: 類別名稱列表，例如 ['background', 'egg', 'broken_egg']
    """
    if class_names is None:
        class_names = ['background', 'class_0', 'class_1']
    
    # 構建 masks 資料夾路徑
    folder_path = Path(folder)
    masks_path = folder_path / 'masks'
    
    if not masks_path.exists():
        raise ValueError(f'masks 資料夾不存在: {masks_path}')
    
    # 獲取所有 mask 圖像
    mask_files = list(masks_path.glob('*.png')) + list(masks_path.glob('*.jpg'))
    
    if len(mask_files) == 0:
        raise ValueError(f'在 {masks_path} 中找不到任何圖像文件')
    
    print(f'找到 {len(mask_files)} 張 mask 圖像')
    print(f'開始分析...\n')
    
    # 分析所有圖像
    all_results = []
    for mask_file in tqdm(mask_files, desc='分析圖像'):
        result = analyze_mask_image(mask_file)
        if result is not None:
            all_results.append(result)
    
    if len(all_results) == 0:
        raise ValueError('沒有成功分析任何圖像')
    
    print(f'\n成功分析 {len(all_results)} 張圖像')
    
    # 計算統計信息
    statistics, total_pixels = calculate_statistics(all_results, class_names)
    
    # 計算權重建議
    loss_weights = calculate_loss_weights(statistics)
    
    # 打印結果
    print_statistics(statistics, total_pixels, loss_weights, class_names)
    
    # 保存結果到 JSON
    output_data = {
        'total_pixels': int(total_pixels),
        'num_images': len(all_results),
        'statistics': statistics,
        'loss_weights': loss_weights
    }
    
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f'統計結果已保存到: {output_path}')
    
    return output_data


def main():
    """主函數（命令行模式）"""
    args = parse_args()
    
    # 使用 run_analysis 函數執行分析
    run_analysis(
        folder=args.folder,
        output=args.output,
        class_names=args.class_names
    )


if __name__ == '__main__':
    # 命令行模式
    #main()
    
    # ============================================================================
    # Spyder 模式：取消註釋下面的代碼，並設定路徑即可在 Spyder 中直接運行
    # ============================================================================
    folder_path = r'Final/train'  # 設定包含 /masks 的資料夾路徑
    output_file = 'class_statistics.json'  # 設定輸出文件
    class_names = ['background', 'egg', 'tip']  # 設定類別名稱
    
    result = run_analysis(folder=folder_path, output=output_file, class_names=class_names)
    # ============================================================================

