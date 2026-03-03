"""Model loading and inference logic for MegaDetector + YOLOv8n-cls."""

import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from PytorchWildlife.models import detection as pw_detection


MEGADETECTOR_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}

DETECTION_IMGSZ = 1280
CLASSIFICATION_IMGSZ = 224
DETECTION_CONF_THRESHOLD = 0.2


class ModelManager:
    """Loads and manages both detection and classification models."""

    def __init__(self, cls_weights: str = "/app/best.pt", device: str = "cuda:0"):
        self.device = device
        self.cls_weights = cls_weights
        self.detector = None
        self.classifier = None
        self._loaded = False

    def load(self):
        """Load both models onto the device."""
        print(f"Loading MegaDetector V6 Compact on {self.device}...")
        self.detector = pw_detection.MegaDetectorV6(device=self.device, pretrained=True, version="yolov10c")
        print("MegaDetector loaded.")

        print(f"Loading YOLOv8n-cls from {self.cls_weights} on {self.device}...")
        self.classifier = YOLO(self.cls_weights)
        print("Classifier loaded.")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, image: Image.Image) -> list[dict]:
        """Run MegaDetector on a single PIL image. Returns list of detection dicts."""
        img_array = np.array(image.convert("RGB"))

        results = self.detector.single_image_detection(
            img=img_array,
            det_conf_thres=DETECTION_CONF_THRESHOLD,
            img_size=DETECTION_IMGSZ,
        )

        detections = []
        if results and "detections" in results:
            det = results["detections"]
            boxes = det.xyxy if hasattr(det, "xyxy") else np.empty((0, 4))
            confs = det.confidence if hasattr(det, "confidence") else np.empty((0,))
            class_ids = det.class_id if hasattr(det, "class_id") else np.empty((0,))

            for i in range(len(boxes)):
                detections.append({
                    "class": MEGADETECTOR_CLASSES.get(int(class_ids[i]), "unknown"),
                    "confidence": round(float(confs[i]), 4),
                    "bbox": [round(float(c), 2) for c in boxes[i]],
                })

        return detections

    def classify(self, image: Image.Image) -> dict:
        """Run YOLOv8n-cls on a single PIL image. Returns classification dict."""
        results = self.classifier.predict(
            source=image,
            imgsz=CLASSIFICATION_IMGSZ,
            device=self.device,
            verbose=False,
        )

        if not results or len(results) == 0:
            return {"top1_class": None, "top1_confidence": 0.0, "top5": []}

        r = results[0]
        probs = r.probs

        top5_indices = probs.top5
        top5_confs = probs.top5conf.cpu().tolist()
        names = r.names

        top5 = [
            {"class": names[idx], "confidence": round(conf, 4)}
            for idx, conf in zip(top5_indices, top5_confs)
        ]

        return {
            "top1_class": names[probs.top1],
            "top1_confidence": round(float(probs.top1conf), 4),
            "top5": top5,
        }

    def process_image(self, image: Image.Image, filename: str) -> dict:
        """Run both models on a single image and return combined result."""
        try:
            detections = self.detect(image)
            classification = self.classify(image)
            return {
                "filename": filename,
                "detections": detections,
                "classifications": [classification],
                "error": None,
            }
        except Exception as e:
            return {
                "filename": filename,
                "detections": [],
                "classifications": [],
                "error": str(e),
            }

    def gpu_info(self) -> dict:
        """Return GPU information."""
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_mem / 1e6, 1),
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
            }
        return {"gpu_available": False}
