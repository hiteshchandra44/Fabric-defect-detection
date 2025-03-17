from ultralytics import YOLO
import cv2

model = YOLO("C:/Users/chand/OneDrive/Desktop/fdd/runs/detect/train4/weights/best.pt")

results = model("C:/Users/chand/Downloads/Samples_25th Feb_2025-20250227T135355Z-001/Samples_25th Feb_2025/Defective piece - Fabric Stain - Without Marking.jpeg", save=True)

