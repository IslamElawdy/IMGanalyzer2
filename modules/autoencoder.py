import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.localization import get_text
from torchvision import transforms
import time
import plotly.express as px

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=32):
        super(Autoencoder, self).__init__()
        # Input: 3x32x32 = 3072
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 32 * 32),
            nn.Sigmoid(),
            nn.Unflatten(1, (3, 32, 32))
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def extract_patches(image_tensor, patch_size=32, stride=32):
    patches = []
    c, h, w = image_tensor.shape
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image_tensor[:, y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return torch.stack(patches)

def render(lang):
    st.header(get_text('module_autoencoder', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']

    st.write(get_text('ae_intro', lang))
    st.info(get_text('math_ae', lang))
    st.latex(r"\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (x_i - \hat{x}_i)^2")

    latent_dim = st.select_slider(get_text('latent_dim', lang), options=[4, 8, 16, 32, 64], value=32)

    w, h = image.size
    new_w = (w // 32) * 32
    new_h = (h // 32) * 32
    if new_w == 0 or new_h == 0:
        st.error("Image too small.")
        return

    img_resized = image.resize((new_w, new_h))
    t_img = transforms.ToTensor()(img_resized)

    patches = extract_patches(t_img)

    st.write(f"Image split into {patches.shape[0]} patches of 32x32 pixels.")

    if 'ae_model' not in st.session_state or st.session_state.get('ae_latent') != latent_dim:
        st.session_state['ae_model'] = Autoencoder(latent_dim)
        st.session_state['ae_latent'] = latent_dim
        st.session_state['ae_trained'] = False

    model = st.session_state['ae_model']

    if st.button(get_text('train_retrain', lang)):
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        progress_bar = st.progress(0)
        loss_text = st.empty()

        dataset = torch.utils.data.TensorDataset(patches)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

        model.train()
        for epoch in range(50):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                optimizer.zero_grad()
                _, output = model(x)
                loss = criterion(output, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if epoch % 5 == 0:
                progress_bar.progress((epoch + 1) / 50)
                loss_text.text(f"Epoch {epoch+1}/50 - Loss: {total_loss / len(loader):.5f}")

        st.session_state['ae_trained'] = True
        st.success(get_text('training_complete', lang))

    if st.session_state.get('ae_trained'):
        model.eval()
        with torch.no_grad():
            _, reconstructed_patches = model(patches)

        recon_image = torch.zeros_like(t_img)
        idx = 0
        patch_size = 32
        stride = 32
        for y in range(0, new_h - patch_size + 1, stride):
            for x in range(0, new_w - patch_size + 1, stride):
                recon_image[:, y:y+patch_size, x:x+patch_size] = reconstructed_patches[idx]
                idx += 1

        c1, c2 = st.columns(2)
        with c1:
            st.subheader(get_text('original', lang))
            st.image(t_img.permute(1,2,0).numpy(), clamp=True, use_container_width=True)

        with c2:
            st.subheader(f"{get_text('reconstructed', lang)} (Latent={latent_dim})")
            st.image(recon_image.permute(1,2,0).numpy(), clamp=True, use_container_width=True)

        diff = (t_img - recon_image).abs().mean(dim=0)
        st.subheader(get_text('recon_error', lang))
        fig = px.imshow(diff.numpy(), color_continuous_scale='hot')
        st.plotly_chart(fig, use_container_width=True)
