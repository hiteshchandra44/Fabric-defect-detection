from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import io
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("C:/Users/chand/OneDrive/Desktop/fdd/runs/detect/train4/weights/best.pt")

# Create response model for detected defects
class DetectionResponse(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

@app.post("/detect/")
async def detect_defect(file: UploadFile = File(...)):
    try:
        # Read image file into memory
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to numpy array (YOLO accepts numpy arrays)
        image_np = np.array(image)

        # Perform inference with YOLOv8
        result = model(image_np)  # Perform detection

        # Extract detection results from the first image in the result (assuming batch size of 1)
        detections = []
        for i in range(len(result.pandas().xyxy[0])):  # Iterate over each detection in the first image
            detection = result.pandas().xyxy[0].iloc[i]
            detection_dict = {
                "x1": detection["xmin"],
                "y1": detection["ymin"],
                "x2": detection["xmax"],
                "y2": detection["ymax"],
                "confidence": detection["confidence"],
                "class_id": detection["class"]
            }
            detections.append(detection_dict)

        # Return the detections as a JSON response
        return {"detections": detections}

    except Exception as e:
        return {"error": str(e)}

# To run the app: uvicorn scripts.app:app --reload
