# YOLOv12 環境設置說明

## 環境設置步驟

### 方法 1: 創建新的 Conda 環境 (推薦)

如果您的 `Lab303` 環境中已有套件可能與 YOLO 訓練衝突，建議創建新的環境：

```bash
# 創建新的 conda 環境
conda create -n YOLO python=3.10 -y

# 激活環境
conda activate YOLO

# 安裝 PyTorch (根據您的 CUDA 版本選擇)
# 如果有 NVIDIA GPU 且已安裝 CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 如果有 NVIDIA GPU 且已安裝 CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 如果只有 CPU 或沒有 CUDA:
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 安裝其他依賴
pip install -r requirements.txt
```

### 方法 2: 在現有 Lab303 環境中安裝

如果您確定不會有衝突，可以直接在現有環境中安裝：

```bash
# 激活現有環境
conda activate Lab303

# 安裝依賴
pip install -r requirements.txt
```

## 驗證安裝

安裝完成後，可以運行以下命令驗證：

```python
python -c "from ultralytics import YOLO; print('Ultralytics 安裝成功')"
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
```

## 訓練腳本使用說明

### 基本使用

```bash
# 使用默認參數訓練
python train.py

# 指定模型和訓練參數
python train.py --model yolo12s-seg.pt --epochs 200 --batch 32 --imgsz 640

# 使用 CPU 訓練
python train.py --device cpu

# 使用特定 GPU
python train.py --device 0
```

### 常用參數說明

- `--model`: 模型大小選擇
  - `yolo12n-seg.pt`: Nano (最小最快)
  - `yolo12s-seg.pt`: Small
  - `yolo12m-seg.pt`: Medium
  - `yolo12l-seg.pt`: Large
  - `yolo12x-seg.pt`: XLarge (最大最準確)

- `--epochs`: 訓練輪數 (建議 100-300)
- `--batch`: 批次大小 (根據 GPU 記憶體調整，建議 8-32)
- `--imgsz`: 輸入圖像尺寸 (建議 640 或 1280)
- `--device`: 訓練設備 (`0`, `1`, `cpu` 等)
- `--lr0`: 初始學習率 (默認 0.01)
- `--data`: 數據集配置文件 (默認 `dataset.yaml`)

### 完整參數示例

```bash
python train.py \
    --model yolo12m-seg.pt \
    --data dataset.yaml \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --lr0 0.01 \
    --workers 8 \
    --name egg_segmentation \
    --project runs/segment
```

## 訓練結果

訓練完成後，結果會保存在 `runs/segment/train/` 目錄下：

- `weights/best.pt`: 最佳模型權重
- `weights/last.pt`: 最後一個 epoch 的權重
- `results.png`: 訓練曲線圖
- `confusion_matrix.png`: 混淆矩陣
- `val_batch0_labels.jpg`: 驗證批次標籤可視化
- `val_batch0_pred.jpg`: 驗證批次預測可視化

## 故障排除

### 1. CUDA 相關錯誤

如果遇到 CUDA 錯誤，請檢查：
- CUDA 版本是否與 PyTorch 版本匹配
- GPU 驅動是否正確安裝
- 可以使用 `--device cpu` 先測試是否為 CUDA 問題

### 2. 記憶體不足

如果遇到 GPU 記憶體不足：
- 減小 `--batch` 參數 (例如從 16 改為 8)
- 減小 `--imgsz` 參數 (例如從 640 改為 416)
- 減少 `--workers` 參數

### 3. 數據集路徑錯誤

確保 `dataset.yaml` 中的路徑正確，可以使用絕對路徑：
```yaml
train: C:/Users/huang/Desktop/egg detection/Final/train/images
val: C:/Users/huang/Desktop/egg detection/Final/val/images
```

## 注意事項

1. 首次運行時，YOLO 會自動下載預訓練模型，請確保網路連線正常
2. 訓練過程會自動保存檢查點，可以隨時中斷並使用 `--resume` 恢復
3. 建議先用小模型 (yolo12n-seg.pt) 測試，確認流程無誤後再使用大模型
4. 訓練時間取決於數據集大小、模型大小和硬體配置

