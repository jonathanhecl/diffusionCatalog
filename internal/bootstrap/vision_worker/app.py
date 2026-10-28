"""
Embedded Python Vision Worker for DiffusionCatalog.
Automatically extracted and managed by the portable catalogd binary.
Provides face detection, face cropping, and 128-d normalized facial embedding extraction.
"""

import os
import sys
from pathlib import Path
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

# Ensure cv2 can find CascadeClassifier if running in an environment with cv2 headless 5.0
try:
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        for sys_path in ["/opt/homebrew/lib/python3.14/site-packages", "/usr/local/lib/python3.14/site-packages", "/opt/homebrew/lib/python3.13/site-packages"]:
            if os.path.exists(sys_path):
                sys.path.insert(0, sys_path)
                break
        import cv2
except Exception:
    pass

import numpy as np

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
    cluster_label: str = "Unassigned"
    expression_tag: str = "Detected"


class DetectResponse(BaseModel):
    faces: List[DetectedFaceOut]


_cached_cascade = None


def get_cascade():
    global _cached_cascade
    if _cached_cascade is not None:
        return _cached_cascade

    # 1. Local directory copy (embedded with catalogd binary)
    local_xml = Path(__file__).parent / "haarcascade_frontalface_default.xml"
    if local_xml.exists() and hasattr(cv2, "CascadeClassifier"):
        c = cv2.CascadeClassifier(str(local_xml))
        if not c.empty():
            _cached_cascade = c
            return c

    # 2. cv2.data.haarcascades
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and hasattr(cv2, "CascadeClassifier"):
        data_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(data_xml):
            c = cv2.CascadeClassifier(data_xml)
            if not c.empty():
                _cached_cascade = c
                return c

    # 3. Known system locations
    if hasattr(cv2, "CascadeClassifier"):
        for p in [
            "/opt/homebrew/lib/python3.14/site-packages/cv2/data/haarcascade_frontalface_default.xml",
            "/opt/homebrew/lib/python3.13/site-packages/cv2/data/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]:
            if os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if not c.empty():
                    _cached_cascade = c
                    return c

    return None


def extract_face_embedding_128(crop_bgr: np.ndarray) -> List[float]:
    """
    Extracts a normalized 128-dimensional facial feature embedding vector
    combining low-frequency DCT structural coefficients (64 dims),
    spatial gradient edge intensities (16 dims), and HSV color distributions (48 dims).
    The resulting vector is L2-normalized so dot product equals cosine similarity.
    """
    try:
        resized = cv2.resize(crop_bgr, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # 1. DCT 8x8 low frequencies (64 dims) - captures structural facial layout
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8].flatten()
        dct_norm = dct_low / (np.linalg.norm(dct_low) + 1e-6)

        # 2. Spatial gradient / Sobel pooled 4x4 (16 dims) - captures facial contours
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx**2 + dy**2)
        pooled_mag = cv2.resize(mag, (4, 4), interpolation=cv2.INTER_AREA).flatten()
        mag_norm = pooled_mag / (np.linalg.norm(pooled_mag) + 1e-6)

        # 3. HSV Color distribution (48 dims: 24 H, 16 S, 8 V) - captures skin & hair tones
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        color = np.concatenate([hist_h, hist_s, hist_v])
        color_norm = color / (np.linalg.norm(color) + 1e-6)

        # 64 (DCT) + 16 (Gradients) + 48 (Color) = 128 dims
        combined = np.concatenate([dct_norm * 0.55, mag_norm * 0.25, color_norm * 0.20])
        total_norm = np.linalg.norm(combined)
        return (combined / (total_norm + 1e-6)).tolist()
    except Exception:
        return [0.0] * 128


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
        img_cv = cv2.imread(req.image_path)
        if img_cv is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            cascade = get_cascade()
            if cascade is not None:
                # Minimum face size: 48x48
                dets = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(48, 48),
                )
                pil_img = Image.open(req.image_path).convert("RGB")
                img_h, img_w = img_cv.shape[:2]

                for idx, (x, y, w, h) in enumerate(dets):
                    # Ensure bbox is within image boundaries
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(img_w, int(x + w))
                    y2 = min(img_h, int(y + h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_cv = img_cv[y1:y2, x1:x2]
                    embedding = extract_face_embedding_128(crop_cv)

                    crop_name = f"asset_{req.asset_id}_face_{idx}.png"
                    crop_abs = crops_dir / crop_name
                    pil_img.crop((x1, y1, x2, y2)).save(crop_abs)

                    # Confidence estimation based on face dimensions and sharpness
                    face_area_ratio = float((w * h)) / float((img_w * img_h) + 1e-6)
                    conf = min(0.99, max(0.65, 0.80 + face_area_ratio * 0.5))

                    faces_out.append(
                        DetectedFaceOut(
                            asset_id=req.asset_id,
                            box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
                            crop_rel_path=f"crops/{crop_name}",
                            confidence=round(conf, 2),
                            embedding=embedding,
                            cluster_label="Unassigned",
                            expression_tag="Detected",
                        )
                    )
    except Exception as e:
        sys.stderr.write(f"[vision_worker] error detecting faces in {req.image_path}: {e}\n")

    return DetectResponse(faces=faces_out)
