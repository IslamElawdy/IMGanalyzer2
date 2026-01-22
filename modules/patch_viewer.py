import streamlit as st
import numpy as np
import plotly.express as px
from utils.localization import get_text

def render(lang):
    st.header(get_text('module_patch', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    img_gray = np.array(image.convert('L'))
    h, w = img_gray.shape

    col1, col2 = st.columns(2)
    with col1:
        patch_size = st.selectbox(get_text('patch_size', lang), [3, 5, 7, 11, 15], index=0)

    half = patch_size // 2

    with col2:
        cx = st.slider(get_text('center_x', lang), half, w - half - 1, w // 2)
        cy = st.slider("Center Y", half, h - half - 1, h // 2)

    x1 = cx - half
    y1 = cy - half
    patch = img_gray[y1:y1+patch_size, x1:x1+patch_size]

    st.write(f"### Patch at ({cx}, {cy})")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**{get_text('visual_heatmap', lang)}**")
        fig = px.imshow(patch, color_continuous_scale='gray', title="Pixel Values")
        fig.update_layout(width=300, height=300)
        st.plotly_chart(fig)

    with c2:
        st.write(f"**{get_text('matrix_values', lang)}**")
        st.dataframe(patch)

    st.markdown("---")
    st.write(f"### {get_text('conv_op_title', lang)}")

    st.info(get_text('math_conv', lang))

    filter_type = st.selectbox(get_text('apply_filter', lang), ["Vertical Edge (Sobel Y)", "Horizontal Edge (Sobel X)", "Average (Blur)"])

    kernel = np.zeros((patch_size, patch_size))

    if filter_type == "Vertical Edge (Sobel Y)":
        for r in range(patch_size):
            for c in range(patch_size):
                if c < half: kernel[r,c] = -1
                elif c > half: kernel[r,c] = 1
    elif filter_type == "Horizontal Edge (Sobel X)":
        for r in range(patch_size):
            for c in range(patch_size):
                if r < half: kernel[r,c] = -1
                elif r > half: kernel[r,c] = 1
    else:
        kernel.fill(1.0 / (patch_size**2))

    k1, k2 = st.columns(2)
    with k1:
        st.write("Kernel (Filter)")
        st.write(kernel)
        fig_k = px.imshow(kernel, color_continuous_scale='RdBu', range_color=[-1,1])
        st.plotly_chart(fig_k, use_container_width=True)

    with k2:
        st.write(get_text('calculation', lang))
        st.latex(r"y = \sum_{i=1}^{k} \sum_{j=1}^{k} I_{i,j} \cdot K_{i,j}")

        element_wise = patch * kernel
        result = np.sum(element_wise)

        st.metric(get_text('conv_result', lang), f"{result:.2f}")

        st.write("Element-wise Multiplication:")
        st.dataframe(element_wise)
