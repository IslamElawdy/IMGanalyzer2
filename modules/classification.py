import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import json
import numpy as np
import cv2
from utils.localization import get_text
from utils.image_processing import to_tensor
import os

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    return model

@st.cache_resource
def load_labels():
    path = os.path.join("utils", "imagenet_classes.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return [f"Class {i}" for i in range(1000)]

def get_gradcam(model, image_tensor, target_class=None):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer = model.features[-1]

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(image_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    score = output[:, target_class]
    score.backward()

    grads = gradients[0].cpu().data.numpy()[0]
    fmaps = activations[0].cpu().data.numpy()[0]

    h1.remove()
    h2.remove()

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(fmaps.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmaps[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (image_tensor.shape[3], image_tensor.shape[2]))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-7)

    return cam, target_class, output

def render(lang):
    st.header(get_text('module_classification', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    st.write(get_text('mobilenet_intro', lang))
    st.info(get_text('math_softmax', lang))
    st.latex(r"P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}")

    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    st.subheader(get_text('top5_pred', lang))
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(5):
        prob = top5_prob[i].item()
        idx = top5_catid[i].item()
        label = labels[idx] if idx < len(labels) else f"Class {idx}"
        st.write(f"**{i+1}. {label}**: {prob*100:.2f}%")
        st.progress(prob)

    st.markdown("---")
    st.subheader(get_text('explainability_title', lang))
    st.write(get_text('gradcam_desc', lang))

    target_idx = st.selectbox(get_text('explain_class', lang),
                              [top5_catid[i].item() for i in range(5)],
                              format_func=lambda x: labels[x] if x < len(labels) else str(x))

    if st.button(get_text('gen_heatmap', lang)):
        try:
            img_tensor_grad = input_tensor.clone().detach().requires_grad_(True)
            cam, _, _ = get_gradcam(model, img_tensor_grad, target_idx)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(image.resize((224, 224)), caption="Input (Crop)", use_container_width=True)
            with c2:
                cam_uint8 = np.uint8(255 * cam)
                heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
            with c3:
                w, h = image.size
                orig_resized = np.array(image.resize((224, 224)))
                if orig_resized.ndim == 2:
                     orig_resized = np.stack([orig_resized]*3, axis=-1)

                overlay = cv2.addWeighted(orig_resized, 0.6, heatmap, 0.4, 0)
                st.image(overlay, caption=get_text('overlay', lang), use_container_width=True)

        except Exception as e:
            st.error(f"Grad-CAM Error: {e}")
