import streamlit as st
from utils.localization import get_text
from utils.image_processing import load_image, normalize_image, to_tensor
import numpy as np
from PIL import Image
import pandas as pd

def render(lang):
    st.header(get_text('module_upload', lang))

    # Upload Section
    uploaded_file = st.file_uploader(get_text('upload_image', lang), type=['jpg', 'jpeg', 'png'])

    # Options
    col1, col2 = st.columns(2)
    with col1:
        # Replaced manual "Full Resolution" checkbox with smart resolution selection below
        pass
    with col2:
        grayscale = st.checkbox(get_text('grayscale', lang), value=False)

    if uploaded_file is not None:
        # First load simply to get dimensions
        try:
            image_initial = Image.open(uploaded_file).convert('RGB')
            w, h = image_initial.size

            # Resolution Selection Logic
            resize_option = "Original"

            if max(w, h) > 1024:
                st.warning(f"High resolution image detected ({w}x{h}). Large images slow down neural network modules.")

                # Options based on aspect ratio
                options = {
                    f"Original ({w}x{h})": None,
                    "Large (Max 1024px)": 1024,
                    "Medium (Max 512px)": 512,
                    "Small (Max 256px)": 256
                }

                selected_label = st.radio("Choose processing resolution:", list(options.keys()), index=1) # Default to 1024
                max_size = options[selected_label]
            else:
                max_size = None # Keep original if small enough

            # Now finalize image
            if max_size:
                ratio = max_size / max(w, h)
                new_size = (int(w * ratio), int(h * ratio))
                image = image_initial.resize(new_size, Image.Resampling.LANCZOS)
            else:
                image = image_initial

        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        if image:
            if grayscale:
                image = image.convert('L')

            st.session_state['uploaded_image'] = image

            # Display Image
            st.image(image, caption=f"Uploaded Image ({image.size[0]}x{image.size[1]})", use_container_width=True)

            # Metadata
            w, h = image.size
            c = len(image.getbands())
            st.info(f"Shape: (H={h}, W={w}, C={c}) | Dtype: uint8 | Range: [0, 255]")

            # Normalization Settings
            norm_mode = st.selectbox(
                get_text('normalization', lang),
                ['[0..1]', '[-1..1]', 'Standardization (mean/std)'],
                index=0
            )

            st.session_state['norm_mode'] = norm_mode

            st.info(get_text('math_norm', lang))
            if norm_mode == '[0..1]':
                st.latex(r"x_{norm} = \frac{x}{255}")
            elif norm_mode == '[-1..1]':
                st.latex(r"x_{norm} = \frac{x}{127.5} - 1")
            elif 'Standardization' in norm_mode:
                st.latex(r"x_{norm} = \frac{x - \mu}{\sigma}")

            # Prepare Tensor for Inspection
            img_np = np.array(image)

            # Map selection to internal code
            mode_map = {'[0..1]': '0-1', '[-1..1]': '-1-1', 'Standardization (mean/std)': 'standard'}
            norm_code = mode_map[norm_mode]

            norm_img = normalize_image(img_np, norm_code)

            st.write(f"### {get_text('tensor_preview', lang)}")
            st.write(f"Min: {norm_img.min():.4f}, Max: {norm_img.max():.4f}, Mean: {norm_img.mean():.4f}, Std: {norm_img.std():.4f}")

            # Example values
            if st.checkbox(get_text('show_raw_values', lang), value=False):
                # Ensure 2D for st.write/dataframe
                if norm_img.ndim == 3:
                    # Crop first
                    crop = norm_img[:5, :5, :]
                    # Format as list of strings
                    df = pd.DataFrame(
                        [[str(np.round(crop[i, j], 2).tolist()) for j in range(crop.shape[1])] for i in range(crop.shape[0])]
                    )
                    st.write(df)
                else:
                    st.write(norm_img[:5, :5])

    else:
        # Optional: Load sample image
        if st.button(get_text('load_sample', lang)):
            # creating a synthetic sample image
            arr = np.zeros((512, 512, 3), dtype=np.uint8)
            # Gradient
            for i in range(512):
                arr[i, :, 0] = i // 2
                arr[:, i, 1] = i // 2
            arr[:, :, 2] = 128

            image = Image.fromarray(arr)
            st.session_state['uploaded_image'] = image
            st.rerun()
