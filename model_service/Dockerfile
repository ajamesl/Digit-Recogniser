FROM python:3.11-slim
WORKDIR /app
COPY model.py mnist_cnn.pth .
RUN pip install torch torchvision fastapi uvicorn
# Make a minimal FastAPI wrapper
COPY api.py .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]