import streamlit as st
import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
import plotly.express as px
from utils.localization import get_text
from utils.image_processing import to_tensor, tensor_to_display

def render(lang):
    st.header(get_text('module_cnn_explainer', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    # Preprocess (Resize for speed and consistent display)
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    img_tensor = t(image) # (C, H, W)

    # Force RGB for consistency
    if img_tensor.shape[0] == 1:
        img_tensor = img_tensor.repeat(3, 1, 1)

    st.markdown(get_text('cnn_expl_intro', lang))

    # Steps
    steps = [
        get_text('step_norm', lang),
        get_text('step_conv', lang),
        get_text('step_relu', lang),
        get_text('step_pool', lang),
        get_text('step_flat', lang),
        get_text('step_cnn', lang)
    ]

    step = st.radio(get_text('nn_step', lang), steps, horizontal=True)

    # 1. Normalization
    if step == get_text('step_norm', lang):
        st.subheader(get_text('step_norm', lang))
        st.markdown(get_text('norm_expl', lang))
        st.info("Code: `image_normalized = image_tensor / 255.0` (Assuming input was 0-255 uint8)")

        # In our app context, `img_tensor` from ToTensor() is already [0, 1].
        # So we explain that raw input (0-255) becomes this.

        c1, c2 = st.columns(2)
        with c1:
            st.write("**Raw Pixel (Virtual)**")
            st.write("255")
        with c2:
            st.write("**Normalized**")
            st.write("1.0")

        st.image(tensor_to_display(img_tensor), caption="Normalized Input", width=300)

    # 2. Convolution
    elif step == get_text('step_conv', lang):
        st.subheader(get_text('step_conv', lang))
        st.markdown(get_text('conv_expl', lang))

        # Interactive Filter
        filter_name = st.selectbox(get_text('select_filter', lang), ["Edge X", "Edge Y", "Sharpen", "Blur"])

        filters = {
            'Edge X': torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32),
            'Edge Y': torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32),
            'Sharpen': torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32),
            'Blur': torch.ones((3, 3), dtype=torch.float32) / 9.0
        }

        k = filters[filter_name]
        st.write(f"**{get_text('kernel_values', lang)}**")
        st.write(k.numpy())

        # Apply to Gray
        gray = img_tensor.mean(dim=0, keepdim=True) # (1, H, W)
        w = k.view(1, 1, 3, 3)
        out = F.conv2d(gray.unsqueeze(0), w, padding=1).squeeze()

        c1, c2 = st.columns(2)
        with c1:
            st.image(tensor_to_display(gray), caption=get_text('original_gray', lang), width=300)
        with c2:
            # Heatmap for negative values
            fig = go.Figure(go.Heatmap(z=out.numpy(), colorscale='RdBu_r', zmid=0))
            fig.update_layout(yaxis=dict(autorange='reversed'), width=300, height=300)
            st.plotly_chart(fig)
            st.caption(get_text('fmap_out', lang))

    # 3. ReLU
    elif step == get_text('step_relu', lang):
        st.subheader(get_text('step_relu', lang))
        st.markdown(get_text('relu_expl', lang))

        # Create an input with negatives (Edge X)
        gray = img_tensor.mean(dim=0, keepdim=True)
        k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3)
        feature_map = F.conv2d(gray.unsqueeze(0), k, padding=1).squeeze()

        relu_out = F.relu(feature_map)

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**{get_text('original_map', lang)}**")
            fig1 = go.Figure(go.Heatmap(z=feature_map.numpy(), colorscale='RdBu_r', zmid=0))
            fig1.update_layout(yaxis=dict(autorange='reversed'), width=300, height=300)
            st.plotly_chart(fig1)
            st.metric(get_text('min_val', lang), f"{feature_map.min().item():.2f}")

        with c2:
            st.write(f"**{get_text('processed_map', lang)} (ReLU)**")
            fig2 = go.Figure(go.Heatmap(z=relu_out.numpy(), colorscale='Gray', zmin=0))
            fig2.update_layout(yaxis=dict(autorange='reversed'), width=300, height=300)
            st.plotly_chart(fig2)
            st.metric(get_text('min_val', lang), f"{relu_out.min().item():.2f}")

    # 4. Pooling
    elif step == get_text('step_pool', lang):
        st.subheader(get_text('step_pool', lang))
        st.markdown(get_text('pool_expl', lang))

        gray = img_tensor.mean(dim=0, keepdim=True)
        pooled = F.max_pool2d(gray.unsqueeze(0), 2).squeeze()

        c1, c2 = st.columns(2)
        with c1:
            st.image(tensor_to_display(gray), caption=f"Original {tuple(gray.shape)}", width=300)
        with c2:
            st.image(tensor_to_display(pooled.unsqueeze(0)), caption=f"Max Pooling 2x2 {tuple(pooled.shape)}", width=150) # Scale visual to show reduction? Streamlit scales automatically.

    # 5. Flattening (New Feature)
    elif step == get_text('step_flat', lang):
        st.subheader(get_text('step_flat', lang))
        st.markdown(get_text('flat_expl', lang))

        st.markdown(f"### {get_text('example_patch', lang)}")

        # Take a tiny patch 4x4
        patch = img_tensor[:, :4, :4] # (3, 4, 4)
        flat = patch.flatten()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{get_text('input_shape', lang)}:** {tuple(patch.shape)}")
            # Show RGB patch zoomed
            fig = px.imshow(tensor_to_display(patch))
            fig.update_layout(width=200, height=200, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig)

        with c2:
            st.markdown(f"**{get_text('output_shape', lang)}:** {tuple(flat.shape)}")
            st.latex(r"\text{Vector} \in \mathbb{R}^{48}")
            st.code(str(flat.numpy().round(1)))

    # 6. Mini-CNN (Recap)
    elif step == get_text('step_cnn', lang):
        st.subheader(get_text('step_cnn', lang))
        st.markdown(get_text('cnn_expl', lang))

        # Run 6 filters
        weights = torch.randn(6, 3, 3, 3) # Random filters
        inp = img_tensor.unsqueeze(0)
        out = F.conv2d(inp, weights, padding=1)

        st.write("6 Random Filters applied to RGB input:")
        cols = st.columns(3)
        for i in range(6):
            with cols[i%3]:
                fm = out[0, i].detach()
                fig = go.Figure(go.Heatmap(z=fm.numpy(), showscale=False))
                fig.update_layout(yaxis=dict(autorange='reversed', visible=False), xaxis=dict(visible=False), margin=dict(l=0,r=0,b=0,t=0), height=150)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Map {i+1}")
