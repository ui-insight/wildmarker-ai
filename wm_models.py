"""Model loading and inference logic for MegaDetector + DINOv3 (ViT-S/16)."""

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.data import transforms as pw_trans


MEGADETECTOR_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}

# DINOv3 classifier (ViT-S/16) fine-tuned on rodant_dataset_v2
DINOV3_CLASSES = ["Back_or_Side", "Chipmunk", "GSqurriel", "Mouse", "Reject"]
DINOV3_IMG_SIZE = 256
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)

DETECTION_CONF_THRESHOLD = 0.2
DETECTION_IMG_SIZE = 960  # multiple of 64 (MDv5 stride); trades recall for speed vs default 1280


class ModelManager:
    """Loads and manages detection (MegaDetector V6) and classification (DINOv3) models."""

    def __init__(self, cls_weights: str = "/app/dinov3_scripted.pt", device: str = "cuda:0"):
        self.device = device
        self.cls_weights = cls_weights
        self.detector = None
        self.classifier = None
        self._loaded = False
        self._cls_transform = T.Compose([
            T.Resize(DINOV3_IMG_SIZE),
            T.CenterCrop(DINOV3_IMG_SIZE),
            T.ToTensor(),
            T.Normalize(mean=DINOV3_MEAN, std=DINOV3_STD),
        ])

    def load(self):
        print(f"Loading MegaDetector V5a on {self.device}...")
        self.detector = pw_detection.MegaDetectorV5(
            device=self.device, pretrained=True, version="a"
        )
        self.detector.IMAGE_SIZE = DETECTION_IMG_SIZE
        self.detector.transform = pw_trans.MegaDetector_v5_Transform(
            target_size=DETECTION_IMG_SIZE, stride=self.detector.STRIDE
        )
        self.detector.model.half()
        _orig_fwd = self.detector.model.forward

        def _half_fwd(x, *a, **kw):
            if x.dtype != torch.float16:
                x = x.half()
            return _orig_fwd(x, *a, **kw)

        self.detector.model.forward = _half_fwd
        print("MegaDetector loaded (FP16).")

        print(f"Loading DINOv3 classifier from {self.cls_weights} on {self.device}...")
        cls = torch.jit.load(self.cls_weights, map_location=self.device).eval()
        cls = cls.half()
        self.classifier = cls
        self._cls_half = True
        print(f"DINOv3 loaded (FP16). Classes: {DINOV3_CLASSES}")

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect(self, image: Image.Image) -> list[dict]:
        img_array = np.array(image.convert("RGB"))
        results = self.detector.single_image_detection(
            img=img_array,
            det_conf_thres=DETECTION_CONF_THRESHOLD,
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

    @torch.inference_mode()
    def classify(self, image: Image.Image) -> dict:
        x = self._cls_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        if getattr(self, "_cls_half", False):
            x = x.half()
        logits = self.classifier(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        k = min(5, probs.shape[0])
        top_conf, top_idx = torch.topk(probs, k=k)
        top_conf = top_conf.cpu().tolist()
        top_idx = top_idx.cpu().tolist()
        top5 = [
            {"class": DINOV3_CLASSES[i] if i < len(DINOV3_CLASSES) else str(i),
             "confidence": round(float(c), 4)}
            for i, c in zip(top_idx, top_conf)
        ]
        return {
            "top1_class": top5[0]["class"],
            "top1_confidence": top5[0]["confidence"],
            "top5": top5,
        }

    def process_image(self, image: Image.Image, filename: str) -> dict:
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
        if torch.cuda.is_available():
            return {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1),
                "gpu_memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1e6, 1),
            }
        return {"gpu_available": False}
