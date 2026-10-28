"""
Embedded Python Vision Worker for DiffusionCatalog.
Automatically extracted and managed by the portable catalogd binary.
Provides multi-scale face detection, face cropping, and 128-d normalized facial embedding extraction.
"""

import os
import sys
from pathlib import Path
from typing import List

# Ensure cv2 can find CascadeClassifier if running in an environment with cv2 headless 5.0
try:
    import cv2
    if not hasattr(cv2, "CascadeClassifier"):
        for sys_path in [
            "/opt/homebrew/lib/python3.14/site-packages",
            "/usr/local/lib/python3.14/site-packages",
            "/opt/homebrew/lib/python3.13/site-packages",
            "/usr/lib/python3/dist-packages",
        ]:
            if os.path.exists(sys_path):
                sys.path.insert(0, sys_path)
                for k in list(sys.modules.keys()):
                    if k == "cv2" or k.startswith("cv2."):
                        del sys.modules[k]
                break
        import cv2
except Exception:
    pass

import cv2
import numpy as np
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
    cluster_label: str = "Unassigned"
    expression_tag: str = "Detected"


class DetectResponse(BaseModel):
    faces: List[DetectedFaceOut]


_cached_cascades = None


def get_cascades():
    global _cached_cascades
    if _cached_cascades is not None:
        return _cached_cascades

    cascades = []
    base_dir = Path(__file__).parent

    xml_names = ["haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml"]
    for name in xml_names:
        # 1. Local directory (embedded in binary)
        p1 = base_dir / name
        if p1.exists() and hasattr(cv2, "CascadeClassifier"):
            c = cv2.CascadeClassifier(str(p1))
            if not c.empty():
                cascades.append(c)
                continue

        # 2. cv2.data.haarcascades
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") and hasattr(cv2, "CascadeClassifier"):
            p2 = Path(cv2.data.haarcascades) / name
            if p2.exists():
                c = cv2.CascadeClassifier(str(p2))
                if not c.empty():
                    cascades.append(c)
                    continue

        # 3. Known system locations
        if hasattr(cv2, "CascadeClassifier"):
            for sys_dir in [
                "/opt/homebrew/lib/python3.14/site-packages/cv2/data",
                "/opt/homebrew/lib/python3.13/site-packages/cv2/data",
                "/usr/share/opencv4/haarcascades",
            ]:
                p3 = Path(sys_dir) / name
                if p3.exists():
                    c = cv2.CascadeClassifier(str(p3))
                    if not c.empty():
                        cascades.append(c)
                        break

    _cached_cascades = cascades
    return cascades


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

        # 1. DCT 8x8 low frequencies (64 dims) - structural facial layout
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8].flatten()
        dct_norm = dct_low / (np.linalg.norm(dct_low) + 1e-6)

        # 2. Spatial gradient / Sobel pooled 4x4 (16 dims) - contours
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(dx**2 + dy**2)
        pooled_mag = cv2.resize(mag, (4, 4), interpolation=cv2.INTER_AREA).flatten()
        mag_norm = pooled_mag / (np.linalg.norm(pooled_mag) + 1e-6)

        # 3. HSV Color distribution (48 dims: 24 H, 16 S, 8 V) - skin & hair tone
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        color = np.concatenate([hist_h, hist_s, hist_v])
        color_norm = color / (np.linalg.norm(color) + 1e-6)

        combined = np.concatenate([dct_norm * 0.55, mag_norm * 0.25, color_norm * 0.20])
        total_norm = np.linalg.norm(combined)
        return (combined / (total_norm + 1e-6)).tolist()
    except Exception:
        return [0.0] * 128


@app.get("/health")
def health():
    cascades = get_cascades()
    return {"status": "ok", "cascades_loaded": len(cascades)}


@app.post("/detect", response_model=DetectResponse)
def detect_faces(req: DetectRequest):
    if not os.path.exists(req.image_path):
        return DetectResponse(faces=[])

    crops_dir = Path(req.cache_dir).resolve() / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    faces_out: List[DetectedFaceOut] = []
    try:
        img_cv = cv2.imread(req.image_path)
        if img_cv is not None:
            img_h, img_w = img_cv.shape[:2]

            # Downscale high-resolution images (> 1200 px) for fast, robust detection
            scale = 1.0
            if max(img_h, img_w) > 1200:
                scale = 1200.0 / float(max(img_h, img_w))
                detect_img = cv2.resize(img_cv, (int(img_w * scale), int(img_h * scale)))
            else:
                detect_img = img_cv

            gray = cv2.cvtColor(detect_img, cv2.COLOR_BGR2GRAY)
            cascades = get_cascades()

            all_raw_boxes = []
            for cascade in cascades:
                dets = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(32, 32),
                )
                for (x, y, w, h) in dets:
                    # Map back to original image coordinates
                    orig_x = max(0, int(float(x) / scale))
                    orig_y = max(0, int(float(y) / scale))
                    orig_w = min(img_w - orig_x, int(float(w) / scale))
                    orig_h = min(img_h - orig_y, int(float(h) / scale))
                    if orig_w >= 48 and orig_h >= 48:
                        all_raw_boxes.append([orig_x, orig_y, orig_w, orig_h])

            if all_raw_boxes:
                # Merge overlapping bounding boxes using Non-Maximum Suppression (NMS)
                scores = [1.0] * len(all_raw_boxes)
                indices = cv2.dnn.NMSBoxes(all_raw_boxes, scores, score_threshold=0.5, nms_threshold=0.35)

                pil_img = Image.open(req.image_path).convert("RGB")
                for count, idx in enumerate(indices):
                    i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                    x, y, w, h = all_raw_boxes[i]

                    # Add a 10% margin around the face for better preview aesthetics
                    margin_x = int(w * 0.10)
                    margin_y = int(h * 0.10)
                    cx1 = max(0, x - margin_x)
                    cy1 = max(0, y - margin_y)
                    cx2 = min(img_w, x + w + margin_x)
                    cy2 = min(img_h, y + h + margin_y)

                    crop_cv = img_cv[cy1:cy2, cx1:cx2]
                    embedding = extract_face_embedding_128(crop_cv)

                    crop_name = f"asset_{req.asset_id}_face_{count}.png"
                    crop_abs = crops_dir / crop_name
                    pil_img.crop((cx1, cy1, cx2, cy2)).save(crop_abs)

                    face_area_ratio = float((w * h)) / float((img_w * img_h) + 1e-6)
                    conf = min(0.99, max(0.70, 0.85 + face_area_ratio * 0.4))

                    faces_out.append(
                        DetectedFaceOut(
                            asset_id=req.asset_id,
                            box=BoundingBox(x=x, y=y, width=w, height=h),
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
