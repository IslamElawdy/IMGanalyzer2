import streamlit as st
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from utils.localization import get_text
from utils.image_processing import to_tensor, tensor_to_display

def add_gaussian_noise(image_np, sigma):
    noise = np.random.normal(0, sigma, image_np.shape)
    noisy = image_np + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_salt_pepper(image_np, prob):
    noisy = np.copy(image_np)
    num_salt = np.ceil(prob * image_np.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image_np.shape]
    noisy[tuple(coords)] = 255

    num_pepper = np.ceil(prob * image_np.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image_np.shape]
    noisy[tuple(coords)] = 0
    return noisy

def jpeg_compression(image_pil, quality):
    buffer = BytesIO()
    image_pil.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer)

def render(lang):
    st.header(get_text('module_noise', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    img_np = np.array(image)

    noise_type = st.selectbox(get_text('noise_type', lang), ["Gaussian", "Salt & Pepper", "Blur", "JPEG Compression"])

    st.info(get_text('math_noise', lang))

    result_image = image

    if noise_type == "Gaussian":
        sigma = st.slider("Sigma (Strength)", 0, 100, 25)
        st.latex(r"I_{noisy} = I + \mathcal{N}(0, \sigma^2)")
        result_np = add_gaussian_noise(img_np, sigma)
        result_image = Image.fromarray(result_np)

    elif noise_type == "Salt & Pepper":
        prob = st.slider("Probability", 0.001, 0.1, 0.01, format="%.3f")
        st.latex(r"P(I_{noisy}=0) = P(I_{noisy}=255) = p/2")
        result_np = add_salt_pepper(img_np, prob)
        result_image = Image.fromarray(result_np)

    elif noise_type == "Blur":
        k = st.slider("Kernel Size", 3, 21, 5, step=2)
        from PIL import ImageFilter
        result_image = image.filter(ImageFilter.GaussianBlur(radius=k/2))

    elif noise_type == "JPEG Compression":
        quality = st.slider("Quality (Lower = more artifacts)", 1, 95, 10)
        result_image = jpeg_compression(image, quality)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader(get_text('original', lang))
        st.image(image, use_container_width=True)

    with c2:
        st.subheader(get_text('distorted', lang))
        st.image(result_image, use_container_width=True)

    with c3:
        st.subheader(get_text('difference', lang))
        res_np = np.array(result_image)
        if res_np.shape != img_np.shape:
             res_np = np.array(result_image.resize(image.size))

        diff = np.abs(img_np.astype(np.float32) - res_np.astype(np.float32))
        if diff.ndim == 3:
            diff = np.mean(diff, axis=2)

        st.image(diff, caption="Absolute Difference (Heatmap)", clamp=True, use_container_width=True)
        st.write(f"Mean Difference: {diff.mean():.2f}")
