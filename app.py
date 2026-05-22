import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import models, transforms

st.set_page_config(page_title="Crop Disease Detection", page_icon="🌱", layout="wide")
st.title("🌱 Crop Disease Detection")
st.markdown("Select one or multiple models to compare predictions on the same image.")

# ── CLASS NAMES (same for all models) ──────────────────────────────
CLASS_NAMES = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# ── MODEL REGISTRY ─────────────────────────────────────────────────
# To add a new model later: add one entry here, done.
MODEL_REGISTRY = {
    "Basic CNN (Keras)": {
        "type": "keras",
        "path": "/content/tomato_model.keras",
        "input_size": 128,
        "normalize": "divide"       # just /255
    },
    "MobileNetV2 (PyTorch)": {
        "type": "pytorch",
        "path": "/content/mobilenet_tomato.pth",
        "input_size": 224,
        "normalize": "imagenet"     # ImageNet mean/std
    },
    "ResNet (PyTorch)": {
        "type": "pytorch",
        "path": "/content/resnet_tomato.pth",
        "input_size": 224,
        "normalize": "imagenet"
    },
    # ── ADD FUTURE MODELS BELOW ────────────────────────────────────
     "AlexNet (PyTorch)": {
         "type": "pytorch",
         "path": "/content/alexnet_tomato.pth",
         "input_size": 224,
         "normalize": "imagenet"
     },
     "DenseNet (PyTorch)": {
         "type": "pytorch",
        "path": "/content/densenet_tomato.pth",
         "input_size": 224,
         "normalize": "imagenet"
     },
     "VGGNet (PyTorch)": {
         "type": "pytorch",
         "path": "/content/vgg_tomato.pth",
         "input_size": 224,
         "normalize": "imagenet"
     },
     "LeNet (PyTorch)": {
         "type": "pytorch",
         "path": "/content/lenet_tomato.pth",
         "input_size": 32,
         "normalize": "divide"
     },
}

# ── PYTORCH ARCHITECTURE BUILDERS ──────────────────────────────────
def build_pytorch_model(name):
    if "MobileNetV2" in name:
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, 10)
    elif "ResNet" in name:
        m = models.resnet50(weights=None)    # change to resnet18/34/101 if needed
        m.fc = nn.Linear(m.fc.in_features, 10)
    elif "AlexNet" in name:
        m = models.alexnet(weights=None)
        m.classifier[6] = nn.Linear(4096, 10)
    elif "DenseNet" in name:
        m = models.densenet121(weights=None) # change to 161/169/201 if needed
        m.classifier = nn.Linear(m.classifier.in_features, 10)
    elif "VGGNet" in name:
        m = models.vgg16(weights=None)       # change to vgg19 if needed
        m.classifier[6] = nn.Linear(4096, 10)
    elif "LeNet" in name:
        # Simple LeNet-5 — must match your training definition exactly
        class LeNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 6, 5), nn.Tanh(), nn.AvgPool2d(2),
                    nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(16*5*5, 120), nn.Tanh(),
                    nn.Linear(120, 84), nn.Tanh(),
                    nn.Linear(84, 10)
                )
            def forward(self, x):
                return self.classifier(self.features(x))
        m = LeNet()
    return m

# ── LAZY MODEL LOADER ───────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    cfg = MODEL_REGISTRY[name]
    if cfg["type"] == "keras":
        return tf.keras.models.load_model(cfg["path"])
    else:
        m = build_pytorch_model(name)
        m.load_state_dict(torch.load(cfg["path"], map_location="cpu"))
        m.eval()
        return m

# ── PREPROCESSING ───────────────────────────────────────────────────
def preprocess(image, name):
    cfg = MODEL_REGISTRY[name]
    size = cfg["input_size"]

    if cfg["type"] == "keras":
        img = np.array(image.resize((size, size))) / 255.0
        return np.expand_dims(img, axis=0)

    if cfg["normalize"] == "imagenet":
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
    return transform(image).unsqueeze(0)

# ── INFERENCE ───────────────────────────────────────────────────────
def predict(name, image):
    model = load_model(name)
    cfg = MODEL_REGISTRY[name]
    tensor = preprocess(image, name)

    if cfg["type"] == "keras":
        preds = model.predict(tensor, verbose=0)[0]
    else:
        with torch.no_grad():
            out = model(tensor)
            preds = torch.softmax(out, dim=1).numpy()[0]

    return preds

# ── SIDEBAR: model selector ─────────────────────────────────────────
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect(
    "Choose models to run",
    options=list(MODEL_REGISTRY.keys()),
    default=["Basic CNN (Keras)"]
)

# ── MAIN: image upload ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.info("Upload an image to get started.")
    st.stop()

if not selected_models:
    st.warning("Select at least one model from the sidebar.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", width=300)
st.divider()

# ── RUN PREDICTIONS ─────────────────────────────────────────────────
results = {}
cols = st.columns(len(selected_models))

for col, name in zip(cols, selected_models):
    with col:
        with st.spinner(f"Running {name}..."):
            preds = predict(name, image)
            top_idx = int(np.argmax(preds))
            top_conf = float(np.max(preds))
            results[name] = {"preds": preds, "top_idx": top_idx, "top_conf": top_conf}

        st.subheader(name)
        st.success(f"**{CLASS_NAMES[top_idx]}**")
        st.metric("Confidence", f"{top_conf:.2%}")

        st.markdown("**All probabilities**")
        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, preds)):
            bar_color = "🟩" if i == top_idx else "⬜"
            short_name = cls.replace("Tomato___", "")
            st.write(f"{bar_color} {short_name}")
            st.progress(float(prob), text=f"{prob:.2%}")

# ── COMPARISON TABLE (only when 2+ models selected) ─────────────────
if len(selected_models) > 1:
    st.divider()
    st.subheader("Comparison Summary")

    # Agreement check
    top_predictions = [results[n]["top_idx"] for n in selected_models]
    all_agree = len(set(top_predictions)) == 1

    if all_agree:
        st.success(f"All models agree: **{CLASS_NAMES[top_predictions[0]].replace('Tomato___', '')}**")
    else:
        st.warning("Models disagree on the prediction.")

    # Table: model vs confidence per class
    import pandas as pd
    table_data = {
        name: [f"{p:.2%}" for p in results[name]["preds"]]
        for name in selected_models
    }
    table_data["Class"] = [c.replace("Tomato___", "") for c in CLASS_NAMES]
    df = pd.DataFrame(table_data).set_index("Class")

    # Highlight the predicted class per model
    st.dataframe(
        df.style.highlight_max(axis=0, color="#d4edda"),
        use_container_width=True
    )
