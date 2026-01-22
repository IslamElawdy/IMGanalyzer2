import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.localization import get_text
from utils.image_processing import normalize_image

def render(lang):
    st.header(get_text('module_pixel', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    img_np = np.array(image)
    h, w = img_np.shape[:2]

    st.markdown(f"### 🔍 {get_text('module_pixel', lang)}")
    st.write(get_text('pixel_intro', lang))
    st.info(get_text('math_pixel', lang))

    # View Mode Selection (Enhancement)
    view_modes = {
        'RGB': 'rgb',
        'Red Channel': 'R',
        'Green Channel': 'G',
        'Blue Channel': 'B',
        'Grayscale': 'gray'
    }
    view_mode_label = st.selectbox(get_text('pixel_view_mode', lang), list(view_modes.keys()))
    view_mode = view_modes[view_mode_label]

    col1, col2 = st.columns(2)
    with col1:
        x_range = st.slider("X Range", 0, w, (0, min(w, 50)), key="px_x")
    with col2:
        y_range = st.slider("Y Range", 0, h, (0, min(h, 50)), key="px_y")

    x1, x2 = x_range
    y1, y2 = y_range

    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1

    # Base Crop
    crop = img_np[y1:y2, x1:x2]
    crop_h, crop_w = crop.shape[:2]

    st.write(f"{get_text('crop_size', lang)}: {crop_w} x {crop_h}")
    show_text = (crop_w <= 32 and crop_h <= 32)

    # Prepare Data based on View Mode
    if view_mode == 'rgb':
        # Default behavior: Show RGB Image
        # If crop is already grayscale (2D), show as gray
        if crop.ndim == 2:
            fig = px.imshow(crop, binary_string=False, color_continuous_scale='gray')
        else:
            fig = px.imshow(crop, binary_string=False)
    else:
        # Single Channel Handling
        if crop.ndim == 3:
            if view_mode == 'R':
                data = crop[:, :, 0]
                cmap = 'Reds'
            elif view_mode == 'G':
                data = crop[:, :, 1]
                cmap = 'Greens'
            elif view_mode == 'B':
                data = crop[:, :, 2]
                cmap = 'Blues'
            elif view_mode == 'gray':
                # Weighted conversion
                data = np.dot(crop[...,:3], [0.299, 0.587, 0.114])
                cmap = 'Gray'
        else:
            # Already 2D
            data = crop
            cmap = 'Gray'

        fig = go.Figure(go.Heatmap(
            z=data,
            colorscale=cmap,
            showscale=True
        ))

        # Add text overlay if small enough
        if show_text:
            text_vals = data.astype(int)
            fig.update_traces(text=text_vals, texttemplate="%{text}")

    # Layout Updates
    fig.update_layout(
        xaxis=dict(showgrid=True, dtick=1, range=[0, crop_w-0.5], visible=False),
        yaxis=dict(showgrid=True, dtick=1, range=[crop_h-0.5, 0], visible=False), # Invert Y for heatmaps
        margin=dict(l=0, r=0, t=0, b=0),
    )
    if view_mode == 'rgb':
        # Image traces don't need explicit Y inversion in layout if px.imshow used
        fig.update_layout(yaxis=dict(visible=False, showgrid=True))

    # Click Event
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", use_container_width=True)

    # Handle Click
    clicked_point = None
    if event and isinstance(event, dict) and 'selection' in event and event['selection']['points']:
        clicked_point = event['selection']['points'][0]
    elif event and hasattr(event, 'selection') and event.selection and event.selection['points']:
        clicked_point = event.selection['points'][0]

    if clicked_point:
        cx = int(clicked_point['x'])
        cy = int(clicked_point['y'])
        orig_x = x1 + cx
        orig_y = y1 + cy

        if 0 <= orig_x < w and 0 <= orig_y < h:
            pixel_val = img_np[orig_y, orig_x] # Always show TRUE RGB value from source

            st.info(f"📍 {get_text('selected_pixel', lang)}: (x={orig_x}, y={orig_y})")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(get_text('original_uint8', lang), str(pixel_val))

            # Normalization (Context: RGB)
            norm_mode = st.session_state.get('norm_mode', '[0..1]')
            mode_map = {'[0..1]': '0-1', '[-1..1]': '-1-1', 'Standardization (mean/std)': 'standard'}
            full_norm = normalize_image(img_np, mode_map[norm_mode])
            norm_val = full_norm[orig_y, orig_x]

            if isinstance(norm_val, np.ndarray) and norm_val.ndim > 0:
                val_str = str(np.round(norm_val, 4).tolist())
            else:
                val_str = str(np.round(norm_val, 4))

            c2.metric(f"{get_text('normalized_val', lang)} ({norm_mode})", val_str)
            c4.metric(get_text('tensor_value_pytorch', lang), val_str)
