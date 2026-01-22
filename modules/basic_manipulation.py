import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from utils.localization import get_text
from utils.image_processing import load_image
from io import BytesIO

# --- MONKEY PATCH START ---
# Fix for streamlit-drawable-canvas incompatibility with Streamlit 1.53+
# st_canvas uses streamlit.elements.image.image_to_url which moved to streamlit.elements.lib.image_utils
try:
    import streamlit.elements.image
    if not hasattr(streamlit.elements.image, 'image_to_url'):
        from streamlit.elements.lib.image_utils import image_to_url
        streamlit.elements.image.image_to_url = image_to_url
except ImportError:
    pass # If we can't patch, we proceed and hope for the best or handle the crash
# --- MONKEY PATCH END ---

# Try-Except block for drawing library
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    st.download_button(
        label=text,
        data=buffered.getvalue(),
        file_name=filename,
        mime="image/png"
    )

def render(lang):
    st.header(get_text('module_manipulation', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    st.write(get_text('manipulation_intro', lang))

    task = st.selectbox("Choose Manipulation",
                        ["File Formats", "Geometric Transformations", "Pixel Operations", "Drawing", "Alpha Blending"])

    if task == "File Formats":
        st.subheader("JPEG vs PNG")
        st.write("JPEG is lossy (creates artifacts), PNG is lossless.")

        quality = st.slider("JPEG Quality", 1, 100, 10)

        # Simulate JPEG
        buffer_jpg = BytesIO()
        image.save(buffer_jpg, "JPEG", quality=quality)
        size_kb = buffer_jpg.getbuffer().nbytes / 1024

        st.image(buffer_jpg, caption=f"JPEG (Quality={quality}, Size={size_kb:.2f} KB)")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download as JPEG", buffer_jpg.getvalue(), f"image_q{quality}.jpg", "image/jpeg")
        with c2:
            buffer_png = BytesIO()
            image.save(buffer_png, "PNG")
            st.download_button("Download as PNG", buffer_png.getvalue(), "image.png", "image/png")

    elif task == "Geometric Transformations":
        st.subheader("Geometry")

        op = st.radio("Operation", ["Resize", "Rotate", "Crop", "Flip"])

        if op == "Resize":
            scale = st.slider("Scale Factor", 0.1, 2.0, 0.5)
            method_str = st.selectbox("Interpolation", ["Nearest", "Bilinear", "Bicubic"])
            method = {"Nearest": Image.NEAREST, "Bilinear": Image.BILINEAR, "Bicubic": Image.BICUBIC}[method_str]

            new_size = (int(image.width * scale), int(image.height * scale))
            res = image.resize(new_size, method)
            st.image(res, caption=f"Resized to {new_size}")
            get_image_download_link(res, "resized.png", "Download Result")

        elif op == "Rotate":
            angle = st.slider("Angle", 0, 360, 45)
            res = image.rotate(angle, expand=True)
            st.image(res, caption=f"Rotated {angle}°")
            get_image_download_link(res, "rotated.png", "Download Result")

        elif op == "Crop":
            st.write("Manual Crop")

            # Interactive Slider Crop with Overlay
            c1, c2 = st.columns(2)
            with c1:
                x = st.slider("X", 0, image.width - 1, 0)
                y = st.slider("Y", 0, image.height - 1, 0)
            with c2:
                w = st.slider("Width", 1, image.width - x, min(100, image.width - x))
                h = st.slider("Height", 1, image.height - y, min(100, image.height - y))

            # Draw overlay on copy
            overlay = image.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([x, y, x+w, y+h], outline="red", width=3)

            st.image(overlay, caption="Crop Preview")

            if st.button("Apply Crop"):
                res = image.crop((x, y, x+w, y+h))
                st.image(res, caption=f"Cropped {w}x{h}")
                get_image_download_link(res, "cropped.png", "Download Result")

        elif op == "Flip":
            mode = st.radio("Mode", ["Horizontal", "Vertical"])
            if mode == "Horizontal":
                res = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                res = image.transpose(Image.FLIP_TOP_BOTTOM)
            st.image(res)
            get_image_download_link(res, "flipped.png", "Download Result")

    elif task == "Pixel Operations":
        st.subheader("Pixel Math")

        op = st.radio("Operation", ["Invert (Negate)", "Grayscale", "Threshold"])

        if op == "Invert (Negate)":
            if image.mode == 'RGBA':
                r,g,b,a = image.split()
                rgb = Image.merge('RGB', (r,g,b))
                res = ImageOps.invert(rgb)
            else:
                res = ImageOps.invert(image.convert('RGB'))
            st.image(res, caption="Inverted")
            st.latex(r"I_{new} = 255 - I_{old}")
            get_image_download_link(res, "inverted.png", "Download Result")

        elif op == "Grayscale":
            res = image.convert('L')
            st.image(res, caption="Grayscale")
            get_image_download_link(res, "grayscale.png", "Download Result")

        elif op == "Threshold":
            thresh = st.slider("Threshold", 0, 255, 128)
            gray = image.convert('L')
            res = gray.point(lambda p: 255 if p > thresh else 0)
            st.image(res, caption="Binary Image")
            st.latex(r"I(x) = \begin{cases} 255 & \text{if } x > T \\ 0 & \text{else} \end{cases}")
            get_image_download_link(res, "threshold.png", "Download Result")

    elif task == "Drawing":
        st.subheader("Interactive Drawing")

        if not CANVAS_AVAILABLE:
            st.error("Library `streamlit-drawable-canvas` not found. Please install it to use this feature.")
        else:
            st.write("Draw on the image using the canvas below.")

            # Resize for canvas if too big
            canvas_width = 600
            ratio = canvas_width / image.width
            new_height = int(image.height * ratio)
            bg_image = image.resize((canvas_width, new_height))

            drawing_mode = st.sidebar.selectbox("Drawing tool:", ("freedraw", "line", "rect", "circle", "transform"))
            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#FF0000")

            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=bg_image,
                update_streamlit=True,
                height=new_height,
                width=canvas_width,
                drawing_mode=drawing_mode,
                key="canvas",
            )

            if canvas_result.image_data is not None:
                # Result is RGBA numpy array
                res_np = canvas_result.image_data
                st.image(res_np, caption="Canvas Result")

                # Convert back to PIL for download
                res_pil = Image.fromarray(res_np.astype('uint8'), 'RGBA')
                get_image_download_link(res_pil, "drawing.png", "Download Result")

    elif task == "Alpha Blending":
        st.subheader("Alpha Blending")
        st.write("Mischung zweier Bilder.")
        st.latex(r"I_{out} = \alpha \cdot I_1 + (1-\alpha) \cdot I_2")

        uploaded_file2 = st.file_uploader("Upload Second Image (Image B)", type=['jpg', 'jpeg', 'png'])

        if uploaded_file2:
            image2 = load_image(uploaded_file2, max_size=1024)
            image2 = image2.resize(image.size)

            alpha = st.slider("Alpha (Opacity of Image A)", 0.0, 1.0, 0.5)

            res = Image.blend(image2, image, alpha)

            c1, c2, c3 = st.columns(3)
            c1.image(image, caption="Image A")
            c2.image(image2, caption="Image B")
            c3.image(res, caption=f"Result (alpha={alpha})")

            get_image_download_link(res, "blended.png", "Download Result")
        else:
            st.info("Bitte lade ein zweites Bild hoch.")
