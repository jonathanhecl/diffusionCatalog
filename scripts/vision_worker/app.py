"""
Embedded Python Vision Worker for DiffusionCatalog.
Automatically extracted and managed by the portable catalogd binary.
Provides state-of-the-art Deep Learning facial detection (YuNet) and
128-d facial recognition embedding extraction (SFace).
"""

import os
import sys
from pathlib import Path
from typing import List

# Ensure cv2 can be cleanly imported
try:
    import cv2
    if not hasattr(cv2, "FaceDetectorYN"):
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


_yunet_detector = None
_sface_recognizer = None


def get_models():
    global _yunet_detector, _sface_recognizer
    if _yunet_detector is not None and _sface_recognizer is not None:
        return _yunet_detector, _sface_recognizer

    base_dir = Path(__file__).parent.resolve()
    yunet_candidates = [
        base_dir / "face_detection_yunet_2023mar.onnx",
        Path("/tmp/yunet.onnx"),
    ]
    sface_candidates = [
        base_dir / "face_recognition_sface_2021dec.onnx",
        Path("/tmp/sface.onnx"),
    ]

    yunet_path = None
    for p in yunet_candidates:
        if p.exists() and p.stat().st_size > 10000:
            yunet_path = str(p)
            break

    sface_path = None
    for p in sface_candidates:
        if p.exists() and p.stat().st_size > 10000:
            sface_path = str(p)
            break

    if yunet_path and hasattr(cv2, "FaceDetectorYN"):
        try:
            _yunet_detector = cv2.FaceDetectorYN.create(
                yunet_path, "", (320, 320), score_threshold=0.60, nms_threshold=0.3, top_k=5000
            )
        except Exception as e:
            sys.stderr.write(f"[vision_worker] warning: could not load YuNet: {e}\n")

    if sface_path and hasattr(cv2, "FaceRecognizerSF"):
        try:
            _sface_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
        except Exception as e:
            sys.stderr.write(f"[vision_worker] warning: could not load SFace: {e}\n")

    return _yunet_detector, _sface_recognizer


@app.get("/health")
def health():
    det, rec = get_models()
    return {
        "status": "ok",
        "has_yunet": det is not None,
        "has_sface": rec is not None,
    }


@app.post("/detect", response_model=DetectResponse)
def detect_faces(req: DetectRequest):
    if not os.path.exists(req.image_path):
        return DetectResponse(faces=[])

    crops_dir = Path(req.cache_dir).resolve() / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    faces_out: List[DetectedFaceOut] = []
    try:
        img_cv = cv2.imread(req.image_path)
        if img_cv is None:
            return DetectResponse(faces=[])

        img_h, img_w = img_cv.shape[:2]
        detector, recognizer = get_models()

        if detector is not None and recognizer is not None:
            # Neural network detection with YuNet
            # Optimal YuNet scale is max dimension ~960px to match its anchor receptive field.
            scale = 1.0
            if max(img_h, img_w) > 960:
                scale = 960.0 / float(max(img_h, img_w))
                det_w = int(img_w * scale)
                det_h = int(img_h * scale)
                det_img = cv2.resize(img_cv, (det_w, det_h))
            else:
                det_img = img_cv
                det_w, det_h = img_w, img_h

            detector.setInputSize((det_w, det_h))
            _, detections = detector.detect(det_img)

            # Secondary pass: if no faces found and image was large, check at higher resolution (1440px)
            if (detections is None or len(detections) == 0) and scale != 1.0 and max(img_h, img_w) > 960:
                scale2 = min(1.0, 1440.0 / float(max(img_h, img_w)))
                det_w2 = int(img_w * scale2)
                det_h2 = int(img_h * scale2)
                det_img2 = cv2.resize(img_cv, (det_w2, det_h2))
                detector.setInputSize((det_w2, det_h2))
                _, dets2 = detector.detect(det_img2)
                if dets2 is not None and len(dets2) > 0:
                    scale = scale2
                    detections = dets2

            if detections is not None and len(detections) > 0:
                pil_img = Image.open(req.image_path).convert("RGB")
                count = 0
                for d in detections:
                    conf = float(d[14])
                    if conf < 0.60:
                        continue

                    # Unscale detection back to original image
                    orig_d = d.copy()
                    if scale != 1.0:
                        orig_d[0] = d[0] / scale
                        orig_d[1] = d[1] / scale
                        orig_d[2] = d[2] / scale
                        orig_d[3] = d[3] / scale
                        for k in range(4, 14):
                            orig_d[k] = d[k] / scale

                    bx = max(0, int(orig_d[0]))
                    by = max(0, int(orig_d[1]))
                    bw = min(img_w - bx, int(orig_d[2]))
                    bh = min(img_h - by, int(orig_d[3]))

                    # Reject tiny artifact patches
                    if bw < 40 or bh < 40:
                        continue

                    # Extract 128-d deep facial embedding with SFace
                    try:
                        aligned = recognizer.alignCrop(img_cv, orig_d)
                        feat = recognizer.feature(aligned).flatten()
                        norm = np.linalg.norm(feat)
                        if norm > 1e-6:
                            feat = feat / norm
                        embedding = [round(float(v), 5) for v in feat]
                    except Exception as fe:
                        sys.stderr.write(f"[vision_worker] sface feature error: {fe}\n")
                        embedding = [0.0] * 128

                    # Aesthetic crop with 20% margin for identity preview
                    margin_x = int(bw * 0.20)
                    margin_y = int(bh * 0.20)
                    cx1 = max(0, bx - margin_x)
                    cy1 = max(0, by - margin_y)
                    cx2 = min(img_w, bx + bw + margin_x)
                    cy2 = min(img_h, by + bh + margin_y)

                    crop_name = f"asset_{req.asset_id}_face_{count}.png"
                    crop_abs = crops_dir / crop_name
                    pil_img.crop((cx1, cy1, cx2, cy2)).save(crop_abs)
                    count += 1

                    faces_out.append(
                        DetectedFaceOut(
                            asset_id=req.asset_id,
                            box=BoundingBox(x=bx, y=by, width=bw, height=bh),
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
