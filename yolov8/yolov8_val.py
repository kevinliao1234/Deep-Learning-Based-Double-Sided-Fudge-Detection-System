from ultralytics import YOLO
from collections import Counter
import os
from tqdm import tqdm

# 載入模型
model = YOLO(r"C:\Users\rem\Desktop\val\v11_model\weights\best.pt")

# 預測資料夾
val_img_dir = r"sample\back_hole"  # 替換為實際路徑

# 蒐集預測類別與未識別圖片
all_preds = []
unidentified_imgs = []  # ⭐️ 沒有辨識結果的圖片
processed_imgs = []     # ⭐️ 有被處理過的圖片（符合圖片格式）

for img_file in tqdm(sorted(os.listdir(val_img_dir))):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(val_img_dir, img_file)
    processed_imgs.append(img_file)

    results = model(img_path)
    boxes = results[0].boxes

    # 沒有預測框，記錄圖片名稱
    if boxes is None or boxes.cls.shape[0] == 0:
        unidentified_imgs.append(img_file)
        continue

    # 取得最高信心度的預測類別
    top_cls = int(boxes.conf.argmax())
    predicted_class = int(boxes.cls[top_cls].item())
    all_preds.append(predicted_class)

# 統計預測為各類別的圖片數量
count = Counter(all_preds)
names = model.names  # 類別名稱對應

print("\n=== 各類別預測數量 ===")
for cls_id in sorted(count.keys()):
    name = names[cls_id] if cls_id in names else f"Class {cls_id}"
    print(f"預測為 {name} 的圖片數量: {count[cls_id]}")

# 顯示未辨識圖片資訊
print(f"\n=== 未辨識圖片（共 {len(unidentified_imgs)} 張） ===")
for name in unidentified_imgs:
    print(name)

# 顯示總處理圖片數與實際圖片總數比對
print(f"\n總處理圖片數: {len(processed_imgs)}")
print(f"未辨識圖片數: {len(unidentified_imgs)}")
print(f"成功辨識圖片數: {len(all_preds)}")