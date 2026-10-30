"""Food recognition demo. Run: python -m streamlit run app.py."""
import json
from pathlib import Path
import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Food Recognition", page_icon="🍽️", layout="wide")

@st.cache_resource
def load_classifier():
    with (ROOT / "food_classes.json").open(encoding="utf-8") as file:
        classes = json.load(file)
    if len(classes) != 101 or len(set(classes)) != 101:
        raise ValueError("Expected 101 unique ordered labels.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(ROOT / "food101_after5epochs.pth", map_location="cpu", weights_only=True))
    model.eval()
    return model, classes

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image, model, classes):
    with torch.inference_mode():
        scores, indices = torch.softmax(model(transform(image).unsqueeze(0)), dim=1)[0].topk(3)
    return [(classes[index], score) for index, score in zip(indices.tolist(), scores.tolist())]

def display_result(source, model, classes):
    try:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Input photo", width=350)
        with right:
            results = predict(image, model, classes)
            st.subheader(results[0][0].replace("_", " ").title())
            st.metric("Model score", f"{results[0][1]:.1%}")
            for label, score in results:
                st.write(f"{label.replace('_', ' ').title()} — {score:.1%}")
                st.progress(score)
            st.caption("Scores are model rankings, not calibrated certainty. Non-food images also receive a food label.")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("Unable to read this image. Try a valid JPG or PNG.")

st.title("Food Recognition")
st.write("Identify a dish from a photo with a fine-tuned ResNet18 model.")
st.caption("Computer vision · 101 categories · PyTorch · CPU inference")
with st.expander("How it works and limitations"):
    st.write("Photo → RGB → 224 × 224 resize → normalization → ResNet18 → top three predictions.")
    st.write("This is image classification, not image or video generation. It does not estimate nutrition or portion size. The training notebook records a best validation accuracy of 76.95%. The bundled demo model has not been evaluated separately; see the README for details.")
try:
    model, classes = load_classifier()
except (OSError, RuntimeError, ValueError, EOFError) as error:
    st.error("Unable to load the model. Check the checkpoint and class label files.")
    st.code(str(error))
    st.stop()

samples_tab, upload_tab, camera_tab = st.tabs(["Try an example", "Upload image", "Use camera"])
with samples_tab:
    samples = sorted((ROOT / "Images_To_Test").glob("*.jpg"))
    if samples:
        sample = st.selectbox("Choose an example", samples, format_func=lambda path: path.stem.replace("_", " ").title())
        display_result(sample, model, classes)
    else:
        st.info("Upload a photo to begin.")
with upload_tab:
    uploaded = st.file_uploader("Choose a food photo", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        display_result(uploaded, model, classes)
with camera_tab:
    photo = st.camera_input("Take a food photo")
    if photo is not None:
        display_result(photo, model, classes)

