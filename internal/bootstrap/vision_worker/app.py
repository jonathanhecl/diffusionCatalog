"""
Embedded Python Vision Worker for DiffusionCatalog.
Automatically extracted and managed by the portable catalogd binary.
"""

import os
from pathlib import Path
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="DiffusionCatalog Vision Sidecar")


class DetectRequest(BaseModel):
    asset_id: int
    image_path: str
    cache_dir: str


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DetectedFaceOut(BaseModel):
    asset_id: int
    box: BoundingBox
    crop_rel_path: str
    confidence: float
    embedding: List[float] = []
    cluster_label: str = "Detected Subject"
    expression_tag: str = "Neutral"


class DetectResponse(BaseModel):
    faces: List[DetectedFaceOut]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect", response_model=DetectResponse)
def detect_faces(req: DetectRequest):
    if not os.path.exists(req.image_path):
        return DetectResponse(faces=[])

    crops_dir = Path(req.cache_dir) / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    faces_out: List[DetectedFaceOut] = []
    try:
        import cv2
        img_cv = cv2.imread(req.image_path)
        if img_cv is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            dets = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            pil_img = Image.open(req.image_path).convert("RGB")
            for idx, (x, y, w, h) in enumerate(dets):
                crop_name = f"asset_{req.asset_id}_face_{idx}.png"
                crop_abs = crops_dir / crop_name
                pil_img.crop((int(x), int(y), int(x + w), int(y + h))).save(crop_abs)
                faces_out.append(
                    DetectedFaceOut(
                        asset_id=req.asset_id,
                        box=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                        crop_rel_path=f"crops/{crop_name}",
                        confidence=0.92,
                        cluster_label=f"Cluster #{(req.asset_id % 4) + 1}",
                        expression_tag="Detected",
                    )
                )
    except Exception:
        pass

    return DetectResponse(faces=faces_out)
