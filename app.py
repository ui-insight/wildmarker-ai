"""FastAPI application for wildlife detection + classification."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

from wm_models import ModelManager, DINOV3_IMG_SIZE, DINOV3_CLASSES


manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.load()
    yield


app = FastAPI(
    title="Wildmarker Detection + Classification Service",
    description="MegaDetector V6 Compact + DINOv3 ViT-S/16 rodent classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Return model status and GPU info."""
    return {
        "status": "ok" if manager.is_loaded else "loading",
        "models": {
            "detection": "MegaDetectorV5a (yolov5x)",
            "classification": "DINOv3 ViT-S/16 (rodent)",
            "classification_classes": DINOV3_CLASSES,
        },
        "gpu": manager.gpu_info(),
    }


@app.post("/predict")
async def predict(files: list[UploadFile] = File(...)):
    """Run detection + classification on 1-2 uploaded images."""
    if not manager.is_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading")

    if len(files) < 1 or len(files) > 2:
        raise HTTPException(status_code=400, detail="Send 1 or 2 image files")

    start = time.perf_counter()
    results = []

    for f in files:
        try:
            contents = await f.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            results.append({
                "filename": f.filename or "unknown",
                "detections": [],
                "classifications": [],
                "error": f"Failed to read image: {e}",
            })
            continue

        result = manager.process_image(image, f.filename or "unknown")
        results.append(result)

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "results": results,
        "metadata": {
            "detection_model": "MegaDetectorV5a (yolov5x)",
            "classification_model": "DINOv3 ViT-S/16 (rodent)",
            "classification_imgsz": DINOV3_IMG_SIZE,
            "device": manager.device,
            "processing_time_ms": elapsed_ms,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
