import streamlit as st
import numpy as np
import pandas as pd
from utils.localization import get_text
import cv2
import plotly.express as px

def get_matrix_latex(matrix):
    """
    Generates a LaTeX bmatrix string for a 2D numpy array.
    """
    if matrix.ndim != 2:
        return ""

    latex_str = r"\begin{bmatrix}"
    for row in matrix:
        latex_str += " & ".join([str(int(x)) for x in row]) + r" \\ "
    latex_str += r"\end{bmatrix}"
    return latex_str

def render(lang):
    st.header(get_text('module_basics', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    img_np = np.array(image) # RGB

    st.write(get_text('basics_intro', lang))

    tabs = st.tabs(["Matrix View", "Color Spaces", "Data Types"])

    with tabs[0]:
        st.subheader("Image as a Matrix")
        st.write("Ein Bild ist ein Gitter aus Zahlen. Hier siehst du einen kleinen Ausschnitt (5x5) als mathematische Matrix.")

        # Center crop 5x5 for Latex readability
        h, w, c = img_np.shape
        cy, cx = h // 2, w // 2
        crop_size = 5
        crop = img_np[cy:cy+crop_size, cx:cx+crop_size]

        st.write(f"Center Crop at ({cx}, {cy})")

        # Show R, G, B matrices as Latex
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Red Channel**")
            # Convert to DataFrame just for inspection if needed, but display as Latex
            latex_r = get_matrix_latex(crop[:,:,0])
            st.latex(latex_r)

        with c2:
            st.write("**Green Channel**")
            latex_g = get_matrix_latex(crop[:,:,1])
            st.latex(latex_g)

        with c3:
            st.write("**Blue Channel**")
            latex_b = get_matrix_latex(crop[:,:,2])
            st.latex(latex_b)

        st.info("Jede Zelle ist ein Pixelwert zwischen 0 (Schwarz) und 255 (Weiß/Volle Farbe).")

    with tabs[1]:
        st.subheader("Color Spaces")
        st.write("Bilder können in verschiedenen Farbräumen dargestellt werden.")

        space = st.selectbox("Select Space", ["RGB", "HSV", "LAB", "YCrCb", "Grayscale"])

        if space == "RGB":
            st.image(image, caption="RGB Image", use_container_width=True)
            st.latex(r"I_{RGB} = [R, G, B]")

        elif space == "HSV":
            st.write("Hue (Farbton), Saturation (Sättigung), Value (Helligkeit).")
            st.latex(r"H \in [0, 179], S \in [0, 255], V \in [0, 255]")
            hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
            c1, c2, c3 = st.columns(3)
            c1.image(hsv[:,:,0], caption="Hue", clamp=True, use_container_width=True)
            c2.image(hsv[:,:,1], caption="Saturation", clamp=True, use_container_width=True)
            c3.image(hsv[:,:,2], caption="Value", clamp=True, use_container_width=True)

        elif space == "LAB":
            st.write("L (Lightness), a (Green-Red), b (Blue-Yellow). Perceptually uniform.")
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            c1, c2, c3 = st.columns(3)
            c1.image(lab[:,:,0], caption="L (Lightness)", clamp=True, use_container_width=True)
            c2.image(lab[:,:,1], caption="A (Green-Red)", clamp=True, use_container_width=True)
            c3.image(lab[:,:,2], caption="B (Blue-Yellow)", clamp=True, use_container_width=True)

        elif space == "YCrCb":
            st.write("Y (Luma), Cr (Red-diff), Cb (Blue-diff). Used in JPEG compression.")
            ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
            c1, c2, c3 = st.columns(3)
            c1.image(ycrcb[:,:,0], caption="Y (Luma)", clamp=True, use_container_width=True)
            c2.image(ycrcb[:,:,1], caption="Cr", clamp=True, use_container_width=True)
            c3.image(ycrcb[:,:,2], caption="Cb", clamp=True, use_container_width=True)

        elif space == "Grayscale":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            st.image(gray, caption="Grayscale", clamp=True, use_container_width=True)
            st.latex(r"Y = 0.299 R + 0.587 G + 0.114 B")

    with tabs[2]:
        st.subheader("Data Types")
        st.write("Computers store images as different data types.")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Integer (uint8)**")
            st.write("Range: [0, 255]. Standard for storage.")
            st.write(img_np.dtype)
            fig = px.histogram(img_np.ravel(), nbins=256, title="Discrete Histogram")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Float (float32)**")
            st.write("Range: [0.0, 1.0]. Standard for Neural Networks.")
            img_float = img_np.astype(np.float32) / 255.0
            st.write(img_float.dtype)
            fig2 = px.histogram(img_float.ravel(), nbins=100, title="Continuous Histogram (0..1)")
            st.plotly_chart(fig2, use_container_width=True)
