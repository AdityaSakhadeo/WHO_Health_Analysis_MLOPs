FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The image is meant for CI/CD packaging; you can run training/inference via:
#   docker run --rm <img> python predictor.py train ...
CMD ["python", "-c", "print('Image built. Run: python predictor.py --help')"]

