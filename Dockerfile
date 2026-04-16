FROM dustynv/l4t-pytorch:r36.4.0

WORKDIR /app

# Install Python dependencies (preserve pre-installed torch/torchvision/numpy from base image)
COPY requirements.txt .
RUN pip freeze | grep -iE '^(torch|torchvision|torchaudio|numpy)==' > /tmp/constraints.txt && \
    pip install --no-cache-dir --index-url https://pypi.org/simple \
    --constraint /tmp/constraints.txt \
    -r requirements.txt

# Copy application code
COPY app.py wm_models.py ./

# Copy DINOv3 ViT-S/16 classifier weights (TorchScript)
COPY weights/dinov3_scripted.pt /app/dinov3_scripted.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
