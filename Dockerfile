FROM python:3.11-slim
WORKDIR /app
COPY app.py . 
RUN pip install streamlit pillow sqlalchemy psycopg2-binary torch torchvision
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]