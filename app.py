from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gradio as gr

MODEL_PATH = "brain_tumor_model.keras"

# Load model
model = load_model(MODEL_PATH)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    processed = preprocess(img)
    preds = model.predict(processed)[0]
    class_index = np.argmax(preds)
    confidence = float(preds[class_index])
    return CLASS_NAMES[class_index], round(confidence, 4)

@app.get("/")
def home():
    return {"message": "Brain Tumor Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    label, conf = predict_image(img)
    return {"prediction": label, "confidence": conf}

# GRADIO DASHBOARD
def gradio_ui(img):
    label, conf = predict_image(img)
    return f"Prediction: {label} (Confidence: {conf})"

demo = gr.Interface(
    fn=gradio_ui,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Brain Tumor Classifier",
    description="Upload an MRI image to classify the tumor.",
)

@app.get("/gradio")
def gradio_app():
    return {"url": "/gradio"}

app = gr.mount_gradio_app(app, demo, path="/dashboard")
