from ultralytics import YOLO

model=YOLO("yolo11n-seg.pt")


model.train(data='yolo_soup.yaml',workers=0,epochs=200,batch=16,imgsz=640,device=0,patience=0) #patience 早停


# model.info()   顯示模型資訊
