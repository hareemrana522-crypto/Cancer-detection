
import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io

CLASSES = ['Adenocarcinoma', 'Benign', 'Large_Cell_Carcinoma',
           'Normal', 'Small_Cell_Carcinoma', 'Squamous_Cell_Carcinoma']

CANCER_INFO = {
    'Adenocarcinoma': '🔴 Most common lung cancer. Starts in mucus-secreting cells.',
    'Benign': '🟢 Non-cancerous growth. Not life-threatening but requires monitoring.',
    'Large_Cell_Carcinoma': '🔴 Fast-growing cancer. Can appear in any part of lung.',
    'Normal': '✅ No cancer detected. Lungs appear healthy.',
    'Small_Cell_Carcinoma': '🔴 Aggressive cancer. Spreads quickly.',
    'Squamous_Cell_Carcinoma': '🟡 Common cancer. Found in central part of lung.'
}

class CrossAttention(nn.Module):
    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, x, context):
        Q, K, V = self.q(x), self.k(context), self.v(context)
        attn = torch.softmax(Q @ K.transpose(-2,-1) * self.scale, dim=-1)
        return self.out(attn @ V)

class LungCancerModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        for param in list(backbone.parameters())[:140]:
            param.requires_grad = False
        self.cnn = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(2048, 512)
        self.cross_attn = CrossAttention(dim=512, heads=8)
        self.norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes))
    def forward(self, x):
        feat = self.cnn(x)
        feat = self.pool(feat).squeeze(-1).squeeze(-1)
        feat = self.proj(feat).unsqueeze(1)
        attn_out = self.cross_attn(feat, feat)
        out = self.norm(feat + attn_out).squeeze(1)
        return self.classifier(self.dropout(out))

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = LungCancerModel(num_classes=6)
    model.load_state_dict(torch.load("best_model_v2.pth", map_location=device))
    model.eval()
    return model, device

def get_gradcam(model, img_tensor):
    gradients, activations = [], []
    def bwd(m, gi, go): gradients.append(go[0])
    def fwd(m, i, o): activations.append(o)
    h1 = model.cnn[-1].register_forward_hook(fwd)
    h2 = model.cnn[-1].register_full_backward_hook(bwd)
    out = model(img_tensor.unsqueeze(0))
    pred = out.argmax(1).item()
    model.zero_grad()
    out[0, pred].backward()
    h1.remove(); h2.remove()
    grad = gradients[0].squeeze().mean(dim=[1,2])
    act = activations[0].squeeze()
    cam = sum(w * act[i] for i, w in enumerate(grad))
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy(), pred

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.set_page_config(page_title="Lung Cancer Detection", page_icon="🫁", layout="wide")

st.markdown("""
<div style="background:linear-gradient(90deg,#1a1a2e,#16213e);
padding:20px;border-radius:10px;text-align:center;margin-bottom:20px;">
<h1 style="color:white;">🫁 Lung Cancer Detection System</h1>
<p style="color:#a0aec0;">Unified Cross-Attention Transformer | Deep Learning AI</p>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Model Accuracy","96.82%")
c2.metric("Test Accuracy","98%")
c3.metric("Training Images","20,764")
c4.metric("Classes","6")

st.markdown("---")
uploaded_file = st.file_uploader("📤 CT Scan Image Upload Karen", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_tensor = transform(image)
    model, device = load_model()

    with st.spinner("AI Analyzing..."):
        img_tensor_d = img_tensor.to(device)
        cam, pred_idx = get_gradcam(model, img_tensor_d)
        with torch.no_grad():
            out = model(img_tensor_d.unsqueeze(0))
            probs = torch.softmax(out, dim=1)[0]
            confidence = probs[pred_idx].item()

    pred_class = CLASSES[pred_idx]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original CT Scan")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Grad-CAM Heatmap")
        cam_r = cv2.resize(cam, (224,224))
        heatmap = cv2.applyColorMap(np.uint8(255*cam_r), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        st.image(heatmap, use_column_width=True)

    with col3:
        st.subheader("Overlay")
        img_np = np.array(image.resize((224,224))).astype(float)/255
        overlay = np.clip(0.6*img_np + 0.4*heatmap/255, 0, 1)
        st.image(overlay, use_column_width=True)

    st.markdown("---")
    if pred_class == "Normal":
        st.success(f"✅ Prediction: {pred_class} | Confidence: {confidence:.1%}")
    else:
        st.error(f"⚠️ Prediction: {pred_class} | Confidence: {confidence:.1%}")

    st.info(CANCER_INFO[pred_class])

    st.subheader("📊 All Class Probabilities")
    for i, cls in enumerate(CLASSES):
        st.progress(float(probs[i]), text=f"{cls}: {probs[i]:.1%}")
