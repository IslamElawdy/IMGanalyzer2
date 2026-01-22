import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.express as px
from utils.localization import get_text
from utils.image_processing import to_tensor, tensor_to_display

def render(lang):
    st.header(get_text('module_conv', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    mode = st.radio(get_text('mode', lang), ["Grayscale", "RGB (Apply to each channel)"], horizontal=True)

    if mode == "Grayscale":
        img_pil = image.convert('L')
        input_tensor = to_tensor(img_pil).unsqueeze(0)
    else:
        img_pil = image
        input_tensor = to_tensor(img_pil).unsqueeze(0)

    st.subheader("Filter Selection")

    filter_choice = st.selectbox(get_text('choose_filter', lang),
                                 ["Identity", "Blur (Average)", "Gaussian Blur", "Sharpen",
                                  "Sobel X (Vertical Edges)", "Sobel Y (Horizontal Edges)",
                                  "Laplacian (All Edges)", "Emboss", "Custom"])

    kernel = None
    k_size = 3

    if filter_choice == "Identity":
        kernel = np.array([[0,0,0], [0,1,0], [0,0,0]], dtype=np.float32)

    elif filter_choice == "Blur (Average)":
        k_size = st.slider("Kernel Size", 3, 15, 3, step=2)
        kernel = np.ones((k_size, k_size), dtype=np.float32) / (k_size**2)
        st.latex(r"K_{i,j} = \frac{1}{k^2}")

    elif filter_choice == "Gaussian Blur":
        k_size = st.slider("Kernel Size", 3, 15, 5, step=2)
        sigma = st.slider("Sigma", 0.1, 5.0, 1.0)
        ax = np.linspace(-(k_size - 1) / 2., (k_size - 1) / 2., k_size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / np.sum(kernel)
        st.latex(r"G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}")

    elif filter_choice == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    elif filter_choice == "Sobel X (Vertical Edges)":
        kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        st.latex(r"G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}")

    elif filter_choice == "Sobel Y (Horizontal Edges)":
        kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
        st.latex(r"G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}")

    elif filter_choice == "Laplacian (All Edges)":
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        st.latex(r"\nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}")

    elif filter_choice == "Emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)

    elif filter_choice == "Custom":
        default_k = "0, -1, 0\n-1, 5, -1\n0, -1, 0"
        txt = st.text_area("Enter Kernel (CSV format)", default_k, height=100)
        try:
            rows = txt.strip().split('\n')
            kernel_list = [[float(x) for x in r.split(',')] for r in rows]
            kernel = np.array(kernel_list, dtype=np.float32)
            k_size = kernel.shape[0]
            if kernel.shape[0] != kernel.shape[1]:
                st.error("Kernel must be square.")
                return
        except:
            st.error("Invalid format")
            return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.write(f"{get_text('kernel_shape', lang)}: {kernel.shape}")
        st.write(kernel)
    with c2:
        fig_k = px.imshow(kernel, color_continuous_scale='RdBu', title=get_text('kernel_heatmap', lang))
        st.plotly_chart(fig_k, use_container_width=True)

    channels = input_tensor.shape[1]
    kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
    kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)

    pad = st.checkbox(get_text('same_padding', lang), value=True)
    padding = k_size // 2 if pad else 0

    try:
        output_tensor = F.conv2d(input_tensor, kernel_tensor, padding=padding, groups=channels)

        st.subheader(get_text('result', lang))

        res_np = tensor_to_display(output_tensor)

        c_in, c_out = st.columns(2)
        with c_in:
            st.image(image, caption=f"Original {input_tensor.shape}", use_container_width=True)
        with c_out:
            st.image(res_np, caption=f"Convolved {output_tensor.shape}", clamp=True, use_container_width=True)

        st.info(get_text('clip_warning', lang))

    except Exception as e:
        st.error(f"Convolution Error: {e}")
