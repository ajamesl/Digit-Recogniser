# Use a slim Python base
FROM python:3.9-slim

# 1) Install dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy the entire repo so subfolders remain intact
WORKDIR /app
COPY . .

# Tell Python to look in /app/model_service when resolving imports
ENV PYTHONPATH=/app/model_service

# 3) Expose ports (8501 for Streamlit; 5000 is internal for Flask)
ENV PORT=8501
EXPOSE 8501 5000

# 4) Start both services:
#    - Run the Flask API in the background (listening on localhost:5000)
#    - Then run Streamlit on $PORT so Render can forward external traffic
CMD bash -c "\
    python model_service/api.py & \
    streamlit run streamlit_app/app.py --server.port \$PORT --server.address 0.0.0.0 \
"
