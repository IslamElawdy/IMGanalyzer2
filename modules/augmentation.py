import streamlit as st
import torch
from torchvision import transforms
from utils.localization import get_text
import numpy as np

def render(lang):
    st.header(get_text('module_augmentation', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    st.write(get_text('aug_intro', lang))
    st.info(get_text('math_aug', lang))

    st.latex(r"\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}")

    st.sidebar.subheader(get_text('active_aug', lang))
    aug_rotation = st.sidebar.checkbox("Random Rotation", value=True)
    aug_flip = st.sidebar.checkbox("Random Horizontal Flip", value=True)
    aug_crop = st.sidebar.checkbox("Random Resized Crop", value=False)
    aug_jitter = st.sidebar.checkbox("Color Jitter", value=True)

    transform_list = []

    if aug_crop:
        transform_list.append(transforms.RandomResizedCrop(size=image.size[::-1], scale=(0.5, 1.0)))
    if aug_rotation:
        transform_list.append(transforms.RandomRotation(degrees=45))
    if aug_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug_jitter:
        transform_list.append(transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))

    pipeline = transforms.Compose(transform_list)

    st.subheader(get_text('aug_preview', lang))

    if st.button(get_text('gen_new_variations', lang)) or True:
        cols = st.columns(4)
        for i in range(8):
            augmented_img = pipeline(image)
            with cols[i % 4]:
                st.image(augmented_img, use_container_width=True, caption=f"Sample {i+1}")

    st.info(get_text('aug_note', lang))
