import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from utils.localization import get_text
from utils.image_processing import to_tensor, tensor_to_display

def render(lang):
    st.header(get_text('module_activation', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    img_tensor = to_tensor(image).unsqueeze(0) # [0, 1]
    img_shifted = (img_tensor * 2) - 1 # [-1, 1]

    st.subheader(get_text('act_relu_title', lang))
    st.write(get_text('relu_desc', lang))
    st.info(get_text('math_relu', lang))
    st.latex(r"ReLU(x) = \max(0, x)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Input (shifted to [-1, 1])")
        st.image(tensor_to_display(img_shifted), caption="Original with Negatives", clamp=True, use_container_width=True)
        st.write(f"Min: {img_shifted.min():.2f}, Max: {img_shifted.max():.2f}")

    relu_out = F.relu(img_shifted)

    with c2:
        st.write(get_text('relu_output', lang))
        st.image(tensor_to_display(relu_out), caption="ReLU Output", clamp=True, use_container_width=True)
        st.write(f"Min: {relu_out.min():.2f}, Max: {relu_out.max():.2f}")

    with c3:
        st.write(get_text('diff_lost', lang))
        diff = img_shifted - relu_out
        st.image(tensor_to_display(diff.abs()), caption="Abs Difference", clamp=True, use_container_width=True)

    st.markdown("---")
    st.subheader(get_text('pooling_title', lang))
    st.info(get_text('math_pooling', lang))

    pool_type = st.radio(get_text('pooling_type', lang), ["Max Pooling", "Average Pooling"], horizontal=True)
    pool_size = st.selectbox("Kernel Size / Stride", [2, 4, 8, 16])

    if pool_type == "Max Pooling":
        pooled = F.max_pool2d(img_tensor, kernel_size=pool_size, stride=pool_size)
        st.latex(r"y_{i,j} = \max_{(p,q) \in \text{Region}_{i,j}} (x_{p,q})")
    else:
        pooled = F.avg_pool2d(img_tensor, kernel_size=pool_size, stride=pool_size)
        st.latex(r"y_{i,j} = \frac{1}{k^2} \sum_{(p,q) \in \text{Region}_{i,j}} x_{p,q}")

    st.write(f"Original Shape: {img_tensor.shape} -> Pooled Shape: {pooled.shape}")
    st.image(tensor_to_display(pooled), width=300, caption=f"Pooled Image ({pool_size}x reduction)")

    st.subheader(get_text('recon_upsample', lang))
    st.write(get_text('recon_desc', lang))

    upsample_mode = st.radio("Upsample Mode", ["nearest", "bilinear"], horizontal=True)

    upsampled = F.interpolate(pooled, size=(img_tensor.shape[2], img_tensor.shape[3]), mode=upsample_mode, align_corners=False if upsample_mode=='bilinear' else None)

    c_orig, c_recon = st.columns(2)
    with c_orig:
        st.image(tensor_to_display(img_tensor), caption="Original", use_container_width=True)
    with c_recon:
        st.image(tensor_to_display(upsampled), caption=f"Restored ({upsample_mode})", use_container_width=True)

    mse = F.mse_loss(img_tensor, upsampled).item()
    st.metric(get_text('info_loss_mse', lang), f"{mse:.6f}")
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(1.0 / mse)
    st.metric("PSNR (dB)", f"{psnr:.2f}")
