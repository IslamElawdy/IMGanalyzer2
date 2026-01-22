import streamlit as st
import torch
import torch.nn as nn
from utils.localization import get_text
from utils.image_processing import to_tensor, tensor_to_display
import numpy as np

class MiniCNN(nn.Module):
    def __init__(self):
        super(MiniCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        f1 = self.conv1(x)
        x = self.relu(f1)
        x = self.pool(x)
        f2 = self.conv2(x)
        return f1, f2

def render(lang):
    st.header(get_text('module_cnn_features', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    st.info(get_text('math_feature', lang))
    st.latex(r"FeatureMap_k = \sigma(\sum_c Input_c * Kernel_{k,c})")

    seed = st.number_input(get_text('random_seed', lang), value=42, step=1)
    torch.manual_seed(seed)

    model = MiniCNN()

    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = t(image).unsqueeze(0)

    st.write(f"{get_text('input_tensor', lang)}: {input_tensor.shape}")

    with torch.no_grad():
        f1, f2 = model(input_tensor)

    st.subheader(get_text('l1_maps', lang))
    st.write(get_text('l1_desc', lang))

    f1_np = f1[0].cpu().numpy()

    cols = st.columns(4)
    for i in range(16):
        with cols[i % 4]:
            fm = f1_np[i]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
            st.image(fm_norm, clamp=True, caption=f"Map {i+1}", use_container_width=True)

    st.markdown("---")
    st.subheader(get_text('l2_maps', lang))
    st.write(get_text('l2_desc', lang))

    f2_np = f2[0].cpu().numpy()
    cols2 = st.columns(4)
    for i in range(8):
        with cols2[i % 4]:
            fm = f2_np[i]
            fm_norm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-9)
            st.image(fm_norm, clamp=True, caption=f"Map {i+1}", use_container_width=True)
