from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="C:/Users/chand/OneDrive/Desktop/fdd/data.yaml", epochs=50, imgsz=640, batch=16)
