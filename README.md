# Wildmarker AI

Dockerized FastAPI service running two wildlife vision models on a Jetson Orin Nano:
- **MegaDetector V6 Compact** (yolov10-c) — animal/person/vehicle detection at 1280px
- **YOLOv8n-cls** (lynx individual ID) — 77-class classification at 224px

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
git clone <repo-url> wildmarker-ai
cd wildmarker-ai
```

Ensure the lynx classifier weights are in place:

```
wildmarker-ai/
  weights/
    best.pt        <-- YOLOv8n-cls lynx individual ID weights
  app.py
  models.py
  Dockerfile
  docker-compose.yml
  ...
```

### Build and start the service

```bash
docker compose build
docker compose up -d
```

The first build pulls the `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3` base image (~8GB) and installs Python dependencies. This takes a while on the first run.

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
        {"class": "animal", "confidence": 0.94, "bbox": [100.0, 200.0, 500.0, 600.0]}
      ],
      "classifications": [
        {
          "top1_class": "LynxID2025_lynx_05",
          "top1_confidence": 0.87,
          "top5": [
            {"class": "LynxID2025_lynx_05", "confidence": 0.87},
            {"class": "LynxID2025_lynx_12", "confidence": 0.05}
          ]
        }
      ],
      "error": null
    }
  ],
  "metadata": {
    "detection_model": "MegaDetectorV6-compact (yolov10-c)",
    "classification_model": "YOLOv8n-cls (lynx, best.pt)",
    "detection_imgsz": 1280,
    "classification_imgsz": 224,
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

- **Base image:** `nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3` (PyTorch 2.5 + CUDA 12.6 pre-installed)
- MegaDetector weights auto-download on first run and are cached in a Docker volume (`model-cache`)
- `best.pt` (lynx classifier) is copied into the container at build time
- Per-image error handling: a corrupt file won't fail the entire request
- The container runs with GPU access via `nvidia-container-runtime`
