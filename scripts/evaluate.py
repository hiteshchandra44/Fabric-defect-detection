from ultralytics import YOLO

model = YOLO("C:/Users/chand/OneDrive/Desktop/fdd/runs/detect/train4/weights/best.pt")

results = model.val()
print(results)
