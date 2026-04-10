import os
import shutil
import random
from pathlib import Path
from collections import Counter
from PIL import Image
from ssd import SSD

# === 輸入參數 ===
n = 200

# 比例設定
if n >= 184:
    x = 0.6
    y = 0.4
else:
    x = 0.8
    y = 0.2

dataset_dir = Path('obj')       # 原始資料集資料夾
output_dir = Path('test_set')   # 測試集輸出資料夾
img_suffix = ['.jpg']           # 圖片副檔名
model_path = r'logs\best_epoch_weights.pth'  # 模型路徑

# === 建立測試資料夾 ===
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True)

# === 隨機抽樣工具 ===
def sample_images(label_folder: str, sample_count: int):
    folder = dataset_dir / label_folder
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in img_suffix]
    sampled = random.sample(image_files, min(sample_count, len(image_files)))
    return sampled

# === 執行抽樣 ===
hole_imgs = sample_images('hole', int(n * x))
normal_imgs = sample_images('normal', int(n * y))
selected_imgs = hole_imgs + normal_imgs

# === 複製圖片與標註 ===
for img_path in selected_imgs:
    label_path = img_path.with_suffix('.txt')
    shutil.copy(img_path, output_dir / img_path.name)
    if label_path.exists():
        shutil.copy(label_path, output_dir / label_path.name)

print(f'✅ 測試集建立完成，共 {len(selected_imgs)} 張圖片。')

# === 載入 YOLOv4 模型 ===
ssd = SSD(model_path=model_path)

# === 統計預測結果（每張圖片計一個主要類別） ===
class_names = ssd.class_names
priority = ['hole', 'leak', 'white', 'normal', 'bhole']  # 類別優先順序
prediction_counter = Counter()

for img_file in output_dir.iterdir():
    if img_file.suffix.lower() not in img_suffix:
        continue
    image = Image.open(img_file)
    _, labels, _ = ssd.detect_image_raw(image)

    labels_set = set(labels)
    for cls_name in priority:
        cls_idx = class_names.index(cls_name)
        if cls_idx in labels_set:
            prediction_counter[cls_name] += 1
            break

# === 顯示統計結果 ===
print('\n📊 預測統計結果（以圖片為單位）：')
for cls_name in ['hole', 'leak', 'white', 'normal']:
    count = prediction_counter.get(cls_name, 0)
    print(f"{cls_name} 出現在 {count} 張圖片中")
