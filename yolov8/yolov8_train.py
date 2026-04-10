from ultralytics import YOLO

model=YOLO("yolov8m-seg.pt")


model.train(data='yolo_soup.yaml',workers=0,epochs=5,batch=16,imgsz=640,device=0,patience=0) #patience 早停


# model.info()   顯示模型資訊
# 'C:\Users\rem\Desktop\ultralytics-main\ultralytics-main\runs\detect\train\weights\best.pt' 原本的權重檔