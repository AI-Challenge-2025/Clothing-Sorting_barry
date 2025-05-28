from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()
model = YOLO("best.pt")  # โหลดโมเดล YOLO ของคุณ

# เปิดให้ frontend เข้าถึง API ได้ทุกที่
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตทุก domain
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()  # อ่านไฟล์รูปภาพที่ส่งมา
    nparr = np.frombuffer(contents, np.uint8)  # แปลงเป็น numpy array
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # decode เป็นรูปภาพ OpenCV

    results = model(img)  # ทำนายด้วยโมเดล YOLO

    detections = []
    # วนลูปดึงข้อมูล detection ที่ได้
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        detections.append({
            "class_name": model.names[int(class_id)],  # ชื่อคลาส
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [x1, y1, x2, y2]
        })

    return {"detections": detections}  # ส่งผลลัพธ์กลับเป็น JSON

# รัน uvicorn เมื่อเรียกไฟล์นี้โดยตรง
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
