# Wildmarker AI

Dockerized FastAPI service running two wildlife vision models on a Jetson Orin Nano:
- **MegaDetector V5a** (YOLOv5x, 140M params) — animal/person/vehicle detection at 960px, FP16
- **DINOv3 ViT-S/16** — 5-class rodent classifier (`Back_or_Side`, `Chipmunk`, `GSqurriel`, `Mouse`, `Reject`) at 256px, FP16

Measured throughput on an Orin Nano Super devkit at 15W: **3.43 img/s** end-to-end (detection + classification), with the GPU saturated. Bump to 25W mode for ~4.2 img/s.

## Quick Start (freshly flashed JetPack 6.x)

If your Orin Nano already has JetPack 6.x flashed and booted, run:

```bash
# Install Docker + NVIDIA runtime
git clone https://github.com/jetsonhacks/install-docker.git
cd install-docker
bash ./install_nvidia_docker.sh
bash ./configure_nvidia_docker.sh
logout  # log back in for docker group to take effect

# Clone and run
git clone https://github.com/ui-insight/wildmarker-ai.git
cd wildmarker-ai
docker compose build
docker compose up -d

# Verify (wait ~45s for models to load)
curl http://localhost:8000/health
```

See the detailed steps below if you need to flash JetPack from scratch.

## Hardware Requirements

- NVIDIA Jetson Orin Nano Developer Kit (8GB)
- microSD card (64GB+ recommended)
- NVMe SSD (optional, recommended for faster I/O)
- USB keyboard, mouse, and display for initial setup
- Ethernet or Wi-Fi connection

## Step 1: Update Firmware (QSPI)

Factory Orin Nano units ship with old firmware that is **not compatible** with JetPack 6.x. You must update the QSPI firmware before installing JetPack 6.

### 1a. Check current firmware version

Power on the Jetson and repeatedly press **Esc** during boot to enter the UEFI menu. The firmware version is shown near the top of the screen. If it already shows **36.x**, skip to Step 2.

### 1b. Flash JetPack 5.1.3 SD card

Download the [JetPack 5.1.3 SD card image](https://developer.nvidia.com/embedded/jetpack-sdk-513) and flash it to your microSD card using [Balena Etcher](https://etcher.balena.io/).

### 1c. Boot and update firmware

Insert the SD card, power on, and complete the initial Ubuntu setup. Then install the QSPI updater and reboot:

```bash
sudo apt-get update
sudo apt-get install nvidia-l4t-jetson-orin-nano-qspi-updater
sudo reboot
```

Watch the firmware update during the reboot. Once it completes, the device will reboot but **will not boot fully** — the firmware is now incompatible with the JetPack 5.1.3 card. Power off the device.

## Step 2: Install JetPack 6.2

Download the [JetPack 6.2 SD card image](https://developer.nvidia.com/embedded/jetpack-sdk-62) and flash it to your microSD card using Balena Etcher.

Insert the card, power on, and complete the initial Ubuntu setup (username, password, locale, etc.).

After first boot, reboot once more to finalize the firmware update:

```bash
sudo reboot
```

### Set power mode

Click the power mode icon in the Ubuntu top bar and select **15W**. This constrains the module to a 15W power envelope, suitable for battery or thermally limited deployments.

You can also set this from the command line:

```bash
sudo nvpmodel -m 0   # 15W mode
sudo jetson_clocks    # pin clocks to max within the 15W budget
```

### Verify

```bash
# Check L4T version (should show R36.4.x)
head -1 /etc/nv_tegra_release

# Check CUDA is available
nvcc --version
```

## Step 3: Install Docker and NVIDIA Container Runtime

JetPack 6.2 SD card images include Docker, but the NVIDIA runtime must be configured. Use the JetsonHacks scripts:

```bash
git clone https://github.com/jetsonhacks/install-docker.git
cd install-docker
bash ./install_nvidia_docker.sh
bash ./configure_nvidia_docker.sh
```

This does three things:
1. Installs/downgrades Docker to a compatible version (27.5.1) to avoid [kernel compatibility issues](https://jetsonhacks.com/2025/02/24/docker-setup-on-jetpack-6-jetson-orin/) with Docker 28.x
2. Adds your user to the `docker` group (log out and back in for this to take effect)
3. Sets `nvidia` as the default Docker runtime in `/etc/docker/daemon.json`

### Verify Docker + GPU access

Log out and back in (or run `newgrp docker`), then:

```bash
docker run --rm nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3 python3 -c "import torch; print(torch.cuda.is_available())"
```

This should print `True`.

## Step 4: Clone and Deploy

```bash
git clone https://github.com/ui-insight/wildmarker-ai.git
cd wildmarker-ai
```

The DINOv3 classifier weights (`weights/dinov3_scripted.pt`, ~83 MB, TorchScript) are committed to the repo. The MegaDetector V5a weights are auto-downloaded on first startup and cached in a Docker volume.

```
wildmarker-ai/
  weights/
    dinov3_scripted.pt   <-- DINOv3 ViT-S/16 rodent classifier
  app.py
  wm_models.py
  Dockerfile
  docker-compose.yml
  ...
```

### Build and start the service

```bash
docker compose build
docker compose up -d
```

The first build pulls the `dustynv/l4t-pytorch:r36.4.0` base image (~8GB) and installs Python dependencies. This takes a while on the first run.

MegaDetector weights are automatically downloaded on first startup and cached in a Docker volume (`model-cache`) so subsequent starts are fast.

The service starts on **port 8000**.

### Check status

```bash
# Container logs
docker compose logs -f

# Health check (wait for models to finish loading)
curl http://localhost:8000/health
```

## API

### `GET /health`

Returns model load status and GPU info.

```bash
curl http://localhost:8000/health
```

### `POST /predict`

Upload 1-2 images for detection + classification.

```bash
# Single image
curl -X POST http://localhost:8000/predict \
  -F "files=@image1.jpg"

# Two images
curl -X POST http://localhost:8000/predict \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
```

**Response:**

```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "detections": [
        {
          "class": "animal",
          "confidence": 0.94,
          "bbox": [100.0, 200.0, 500.0, 600.0],
          "classification": {
            "top1_class": "GSqurriel",
            "top1_confidence": 0.87,
            "top5": [
              {"class": "GSqurriel", "confidence": 0.87},
              {"class": "Chipmunk", "confidence": 0.08}
            ]
          }
        }
      ],
      "error": null
    }
  ],
  "metadata": {
    "detection_model": "MegaDetectorV5a (yolov5x)",
    "classification_model": "DINOv3 ViT-S/16 (rodent)",
    "classification_imgsz": 256,
    "device": "cuda:0",
    "processing_time_ms": 450.2
  }
}
```

## Testing

With the service running:

```bash
pip install requests
python3 test_service.py
```

The test script sends images from `test_images/` and validates detection/classification output plus error handling.

## Troubleshooting

**"Models are still loading" (503):** The models take 30-60 seconds to load on first startup. Wait and retry `/health`.

**Docker permission denied:** Log out and back in after running the Docker install scripts, or use `newgrp docker`.

**Out of memory:** The Orin Nano has 8GB shared CPU/GPU memory. Close other applications. Running in 15W mode reduces available GPU frequency but does not affect memory capacity.

**Docker build fails with CUDA errors:** Ensure the NVIDIA runtime is set as default. Check with:
```bash
cat /etc/docker/daemon.json
# Should contain: "default-runtime": "nvidia"
```

## Architecture

- **Base image:** `dustynv/l4t-pytorch:r36.4.0` (PyTorch 2.4 + CUDA 12.6 pre-installed)
- **Detection:** MegaDetector V5a (YOLOv5x) loaded via PytorchWildlife, weights auto-downloaded on first run and cached in a Docker volume (`model-cache`). Input fixed at 960×960, FP16 inference.
- **Classification:** DINOv3 ViT-S/16 as a TorchScript-traced graph (`weights/dinov3_scripted.pt`, ~83 MB), copied into the container at build time. Input 256×256, FP16 inference. The classifier runs on the **bbox crop** of each `class=animal` detection; `person` and `vehicle` detections are returned without a classification.
- Per-image error handling: a corrupt file won't fail the entire request
- The container runs with GPU access via `nvidia-container-runtime`

### Performance notes

- Paired top+front camera requests (two images per POST) at client concurrency 2 saturate the GPU and deliver **3.43 img/s at 15W** / **4.19 img/s at 25W** on the Orin Nano Super devkit.
- The GPU is the bottleneck (p50 99% util); faster throughput at the same resolution requires TensorRT export rather than a power-mode change.
