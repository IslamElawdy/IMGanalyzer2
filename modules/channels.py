import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.localization import get_text

def render(lang):
    st.header(get_text('module_channels', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    img_np = np.array(image)

    # Check if Grayscale or RGB
    if len(img_np.shape) == 2:
        st.write(get_text('image_grayscale', lang))
        st.image(img_np, caption="Grayscale Channel", clamp=True, width=300)

        fig = px.histogram(img_np.ravel(), nbins=256, title="Grayscale Histogram")
        st.plotly_chart(fig, use_container_width=True)
        return

    h, w, c = img_np.shape
    st.write(f"Image Shape: {h}x{w}x{c} (RGB)")
    st.markdown(get_text('color_stack_intro', lang))

    st.info(get_text('math_tensor', lang))
    st.latex(r"T \in \mathbb{R}^{C \times H \times W}")

    # Decomposition
    tabs = st.tabs([get_text('separation', lang), get_text('3d_view', lang), get_text('histograms', lang)])

    with tabs[0]:
        c1, c2, c3 = st.columns(3)

        # Red
        r_channel = img_np[:, :, 0]
        c1.write("### Red Channel")
        c1.image(r_channel, clamp=True, use_container_width=True)
        # Heatmap toggle
        if c1.checkbox("Show R-Heatmap"):
            fig_r = px.imshow(r_channel, color_continuous_scale='Reds')
            c1.plotly_chart(fig_r, use_container_width=True)

        # Green
        g_channel = img_np[:, :, 1]
        c2.write("### Green Channel")
        c2.image(g_channel, clamp=True, use_container_width=True)
        if c2.checkbox("Show G-Heatmap"):
            fig_g = px.imshow(g_channel, color_continuous_scale='Greens')
            c2.plotly_chart(fig_g, use_container_width=True)

        # Blue
        b_channel = img_np[:, :, 2]
        c3.write("### Blue Channel")
        c3.image(b_channel, clamp=True, use_container_width=True)
        if c3.checkbox("Show B-Heatmap"):
            fig_b = px.imshow(b_channel, color_continuous_scale='Blues')
            c3.plotly_chart(fig_b, use_container_width=True)

    with tabs[1]:
        st.subheader("Tensor View: (C, H, W)")
        st.write("In Deep Learning libraries like PyTorch, we often move the Channel dimension to the front.")

        st.write(get_text('stack_viz_intro', lang))

        patch_x = np.random.randint(0, w-10)
        patch_y = np.random.randint(0, h-10)
        patch = img_np[patch_y:patch_y+10, patch_x:patch_x+10] # 10x10x3

        X, Y, Z, Val = [], [], [], []
        colors = []

        for z_idx, color, z_pos in [(0, 'red', 0), (1, 'green', 1), (2, 'blue', 2)]:
            for r in range(10):
                for c in range(10):
                    X.append(c)
                    Y.append(10-r)
                    Z.append(z_pos)
                    v = patch[r, c, z_idx]
                    Val.append(v)
                    colors.append(f'rgb({v},0,0)' if z_idx==0 else (f'rgb(0,{v},0)' if z_idx==1 else f'rgb(0,0,{v})'))

        fig_3d = go.Figure(data=[go.Scatter3d(
            x=X, y=Y, z=Z,
            mode='markers',
            marker=dict(size=5, color=colors, opacity=0.8)
        )])
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Width (x)',
                yaxis_title='Height (y)',
                zaxis_title='Channel (c)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with tabs[2]:
        st.subheader(get_text('color_dist', lang))

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(x=r_channel.ravel(), name='Red', marker_color='red', opacity=0.6))
        fig_hist.add_trace(go.Histogram(x=g_channel.ravel(), name='Green', marker_color='green', opacity=0.6))
        fig_hist.add_trace(go.Histogram(x=b_channel.ravel(), name='Blue', marker_color='blue', opacity=0.6))

        fig_hist.update_layout(barmode='overlay')
        fig_hist.update_traces(opacity=0.5)
        st.plotly_chart(fig_hist, use_container_width=True)
