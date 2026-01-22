import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from utils.localization import get_text
from modules.classification import load_model, load_labels
from utils.image_processing import tensor_to_display

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

def render(lang):
    st.header(get_text('module_adversarial', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    st.write(get_text('fgsm_intro', lang))
    st.warning(get_text('edu_warning', lang))
    st.info(get_text('math_fgsm', lang))
    st.latex(r"x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))")

    try:
        model = load_model()
        labels = load_labels()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    epsilon = st.slider("Epsilon (Perturbation Strength)", 0.0, 0.3, 0.05, step=0.01)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    input_tensor = preprocess(image).unsqueeze(0)
    input_tensor.requires_grad = True

    output = model(input_tensor)
    init_pred = output.max(1, keepdim=True)[1]

    init_label = labels[init_pred.item()] if init_pred.item() < len(labels) else str(init_pred.item())
    st.write(f"{get_text('orig_pred', lang)}: **{init_label}**")

    loss = nn.CrossEntropyLoss()(output, init_pred[0])

    model.zero_grad()
    loss.backward()
    data_grad = input_tensor.grad.data

    perturbed_data = fgsm_attack(input_tensor, epsilon, data_grad)

    output_adv = model(perturbed_data)
    adv_pred = output_adv.max(1, keepdim=True)[1]
    adv_label = labels[adv_pred.item()] if adv_pred.item() < len(labels) else str(adv_pred.item())

    c1, c2, c3 = st.columns(3)

    with c1:
        st.write(get_text('original', lang))
        orig_disp = inv_normalize(input_tensor[0]).detach().cpu().numpy().transpose(1,2,0)
        orig_disp = np.clip(orig_disp, 0, 1)
        st.image(orig_disp, use_container_width=True)

    with c2:
        st.write(get_text('noise_x50', lang))
        noise = data_grad[0].sign().cpu().numpy().transpose(1,2,0)
        noise_disp = (noise + 1) / 2
        st.image(noise_disp, use_container_width=True)

    with c3:
        st.write(f"Adversarial (Pred: {adv_label})")
        adv_disp = inv_normalize(perturbed_data[0]).detach().cpu().numpy().transpose(1,2,0)
        adv_disp = np.clip(adv_disp, 0, 1)
        st.image(adv_disp, use_container_width=True)

    if init_pred.item() != adv_pred.item():
        st.error(f"{get_text('attack_success', lang)} {init_label} -> {adv_label}")
    else:
        st.success(get_text('attack_failed', lang))
