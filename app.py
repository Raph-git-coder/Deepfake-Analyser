import streamlit as st
from transformers import ViTForImageClassification
from PIL import Image
import torch
from torchvision import transforms
import os
try:
    import numpy as np
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
    import numpy as np

# ------------------------------
# 🎨 Streamlit Page Config
# ------------------------------
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# 🌈 Custom CSS Styling
# ------------------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #111827, #1f2937);
            color: white;
        }
        h1, h2, h3, h4 {
            text-align: center;
            color: #f3f4f6;
        }
        .uploadedImage {
            border-radius: 12px;
            border: 2px solid #4ade80;
            padding: 5px;
        }
        .prediction-card {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-top: 20px;
        }
        .confidence {
            font-size: 1.2rem;
            color: #9ca3af;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# ⚙️ Load Model
# ------------------------------
MODEL_PATH = "vit_deepfake.pt"

@st.cache_resource
def load_model():
    try:
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            return model
        else:
            st.warning("⚠️ Model file not found! Please upload vit_deepfake.pt in the same directory.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ------------------------------
# 🧠 App Header
# ------------------------------
st.title("🎭 Deepfake Detection using Vision Transformer")
st.markdown(
    "Upload a face image to detect whether it’s **Real** 🟢 or **Fake** 🔴. "
    "Powered by a fine-tuned ViT (Vision Transformer)."
)

# ------------------------------
# 📂 File Uploader
# ------------------------------
uploaded_file = st.file_uploader("📸 Choose an image...", type=["jpg", "jpeg", "png"])

# ------------------------------
# 🧩 Sidebar Info
# ------------------------------
with st.sidebar:
    st.header("ℹ️ About the App")
    st.write("""
    - **Model:** ViT (Vision Transformer)
    - **THIS IS JUST AN WORKING PROTOTYPE**
    - **Framework:** Hugging Face Transformers
    - **Interface:** Streamlit
    - **Output:** Real 🟢 or Fake 🔴
    """)
    st.markdown("---")
    st.write("👨‍💻 Created with ❤️ using Streamlit & PyTorch")
    st.write("Created by -**Raphael.N** of Grade 11")

# ------------------------------
# 🚀 Process Image
# ------------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

        if model is not None:
            with st.spinner("🔍 Analyzing image..."):
                # Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5]),
                ])
                image_tensor = transform(image).unsqueeze(0)

                # Inference
                with torch.no_grad():
                    outputs = model(pixel_values=image_tensor)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                    pred_class = torch.argmax(probs).item()
                    confidence = probs[pred_class].item() * 100

                label = "🟢 Real" if pred_class == 0 else "🔴 Fake"
                emoji = "✅" if pred_class == 0 else "⚠️"

                # Display Result
                st.markdown(f"""
                    <div class="prediction-card">
                        <h2>{emoji} Prediction: {label}</h2>
                        <p class="confidence">Confidence: {confidence:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Model not loaded. Please check your model file path.")

    except Exception as e:
        st.error(f"⚠️ Error processing the image: {e}")
