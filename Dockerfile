FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch
RUN pip install --no-cache-dir --default-timeout=300 --retries=10 \
    "https://download.pytorch.org/whl/cpu/torch-2.14.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"

# Install application dependencies from the standard PyPI HTML index
RUN pip install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    --default-timeout=300 \
    --retries=10 \
    --prefer-binary \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]