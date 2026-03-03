FROM nvcr.io/nvidia/l4t-pytorch:r36.4.0-pth2.5-py3

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py models.py ./

# Copy lynx classifier weights
COPY weights/best.pt /app/best.pt

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
