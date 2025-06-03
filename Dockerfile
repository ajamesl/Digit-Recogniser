# --------------- Stage 0: common base with dependencies ---------------
FROM python:3.9-slim AS base

# Set the root of your project inside the container
WORKDIR /app

# Copy only the requirements file from the host into /app
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt


# --------------- Stage 1: build model_service image ---------------
FROM base AS model_service

# Switch to the model_service subdirectory
WORKDIR /app/model_service

# Copy everything under Digit-Recogniser/model_service/ into /app/model_service
COPY model_service/ .

# (If your model .pth or other assets live outside model_service/, copy them here as well:
#  e.g. COPY mnist_cnn.pth /app/model_service/  )

# Expose whichever port your API listens on (e.g. 5000)
EXPOSE 5000

# Default command to run your API server
CMD ["python", "api.py"]


# --------------- Stage 2: build streamlit_app image ---------------
FROM base AS streamlit_app

# Switch to the streamlit_app subdirectory
WORKDIR /app/streamlit_app

# Copy everything under Digit-Recogniser/streamlit_app/ into /app/streamlit_app
COPY streamlit_app/ .

# Expose Streamlit’s default port
EXPOSE 8501

# Command to launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
