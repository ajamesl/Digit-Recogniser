import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import torch
- from model import MNIST_CNN
+ from model_service.model import MNIST_CN
import io
from datetime import datetime
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData

# 1) Sidebar: database URL
DB_URL = st.sidebar.text_input(
    "Postgres URL",
    "postgresql://Arjuna@localhost:5432/mnist_logs"
)

# 2) Connect to DB
engine = create_engine(DB_URL)
metadata = MetaData()
logs = Table('logs', metadata,
             Column('id', Integer, primary_key=True),
             Column('timestamp', String),
             Column('predicted', Integer),
             Column('true_label', Integer))
metadata.create_all(engine)

# 3) Load model once
@st.cache_resource
def load_model():
    m = MNIST_CNN()
    m.load_state_dict(torch.load('mnist_cnn.pth', map_location='cpu'))
    m.eval()
    return m

model = load_model()

st.title("MNIST Digit Classifier")
canvas = st.empty()

# 4) Drawing canvas
from streamlit_drawable_canvas import st_canvas

canvas_result = st_canvas(
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
    img = img.resize((28, 28))
    # ...continue with tensor conversion and predictio

if img is not None:
    # Preprocess
    #img = Image.open(io.BytesIO(img)).convert('L').resize((28,28))
    img = img.convert('L').resize((28, 28))
    arr = np.array(img)/255.0
    tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(0).float()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).numpy().flatten()
        pred = int(probs.argmax())
        conf = float(probs.max())
    st.write(f"**Prediction:** {pred} ({conf:.2%} confidence)")

    # True label entry
    true = st.number_input("True label", min_value=0, max_value=9, step=1)
    if st.button("Submit"):
       ins = logs.insert().values(
          timestamp=datetime.utcnow().isoformat(),
          predicted=pred,
          true_label=int(true)
     )
       # use a connection/transaction context instead of engine.execute()
       with engine.begin() as conn:
          conn.execute(ins)
       st.success("Logged to database")

    # if st.button("Submit"):
    #     ins = logs.insert().values(
    #         timestamp=datetime.utcnow().isoformat(),
    #         predicted=pred,
    #         true_label=int(true)
    #     )
    #     engine.execute(ins)
    #     st.success("Logged to database")
