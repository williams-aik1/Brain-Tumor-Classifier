from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
import gradio as gr
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

MODEL_PATH = "brain_tumor_model.keras"

# Load saved model
model = load_model(MODEL_PATH)

CLASS_NAMES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

app = FastAPI()

# CORS for public access
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
    return CLASS_NAMES[class_index], confidence

# ---------------------------
#  GRAD-CAM IMPLEMENTATION
# ---------------------------

def generate_gradcam(img, layer_name=None):
    img_array = preprocess(img)

    # Pick last conv layer automatically
    if layer_name is None:
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv layer has 4D output
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0,1))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize 0–1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    heatmap = np.uint8(255 * heatmap)

    # Resize heatmap to original image
    heatmap_img = Image.fromarray(heatmap).resize(img.size)
    heatmap_img = heatmap_img.convert("RGB")

    # Apply color map
    heatmap_img = np.array(heatmap_img)
    heatmap_img = np.uint8(255 * heatmap_img / np.max(heatmap_img))
    
    return heatmap_img

# ---------------------------
#  API ENDPOINTS
# ---------------------------

@app.get("/")
def home():
    return {"message": "Brain Tumor Classifier API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    label, conf = predict_image(img)
    return {"prediction": label, "confidence": conf}

# ---------------------------
#  GRADIO DASHBOARD
# ---------------------------

def gradio_ui(img):
    label, conf = predict_image(img)
    heatmap = generate_gradcam(img)

    return (
        f"Prediction: {label} (Confidence: {round(conf,4)})",
        heatmap
    )

demo = gr.Interface(
    fn=gradio_ui,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Textbox(label="Prediction Result"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="🧠 Brain Tumor Classifier (Educational Use Only)",
    description="""
### Upload an MRI image to classify the tumor type.

This tool predicts **Glioma, Meningioma, Pituitary**, or **No Tumor**  
and shows a **Grad-CAM heatmap** of the region influencing the model.

⚠️ **DISCLAIMER:**  
*This tool is for strictly educational and research purposes only.  
It is NOT a medical diagnostic tool and must NOT be used for clinical decisions.*
""",
    allow_flagging="never",
    theme="soft"
)

# GRADIO MOUNT FIX
from gradio.routes import mount_gradio_app

@app.get("/gradio")
def gradio_app():
    return {"url": "/dashboard"}

mount_gradio_app(app, demo, path="/dashboard")

