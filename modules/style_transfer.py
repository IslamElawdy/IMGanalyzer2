import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import copy
import numpy as np
from utils.localization import get_text
from utils.image_processing import load_image

@st.cache_resource
def load_vgg():
    # VGG19 features only
    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
    return cnn

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class ContentLoss(nn.Module):
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

def get_style_model_and_losses(cnn, style_img, content_img, content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']):
    normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    normalization_std = torch.tensor([0.229, 0.224, 0.225])

    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

def render(lang):
    st.header(get_text('module_style', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    content_image_pil = st.session_state['uploaded_image']
    st.write(get_text('style_intro', lang))
    st.info(get_text('math_style', lang))
    st.latex(r"\mathcal{L}_{total} = \alpha \mathcal{L}_{content} + \beta \mathcal{L}_{style}")

    # Style Image Input
    st.subheader("Style Image")
    style_source = st.radio("Source", ["Upload", "Generate Random Pattern"])

    style_image_pil = None
    if style_source == "Upload":
        f = st.file_uploader("Upload Style Image", type=['jpg', 'png'])
        if f:
            style_image_pil = load_image(f, max_size=256) # Resize small for speed
    else:
        # Generate random noise/pattern
        if st.button("Generate Random Style"):
             arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
             # Smooth it a bit
             from PIL import ImageFilter
             style_image_pil = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(2))
             st.session_state['generated_style'] = style_image_pil
        elif 'generated_style' in st.session_state:
             style_image_pil = st.session_state['generated_style']

    if style_image_pil:
        st.image(style_image_pil, caption="Style Image", width=200)

    if st.button("Start Style Transfer (Slow on CPU)"):
        if not style_image_pil:
            st.error("Please provide a style image.")
            return

        try:
            cnn = load_vgg()
        except Exception as e:
            st.error(f"Error loading VGG: {e}")
            return

        # Resize content to match small size for speed
        imsize = 256 # larger = slower
        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.CenterCrop(imsize),
            transforms.ToTensor()
        ])

        style_img = loader(style_image_pil).unsqueeze(0)
        content_img = loader(content_image_pil).unsqueeze(0)
        input_img = content_img.clone()

        st.write("Building the Style Transfer Model...")
        model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

        optimizer = optim.LBFGS([input_img.requires_grad_()])

        st.write("Optimizing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        display_area = st.empty()

        run = [0]
        while run[0] <= 50: # Limit steps
            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= 1000000
                content_score *= 1

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 10 == 0:
                    status_text.text(f"Step {run[0]}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")
                    progress_bar.progress(min(run[0]/50, 1.0))

                    # Live preview
                    display_area.image(input_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(), clamp=True, width=300)

                return loss

            optimizer.step(closure)

        # Final Correction
        input_img.data.clamp_(0, 1)
        st.success("Transfer Complete!")
        st.image(input_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(), caption="Styled Image", use_container_width=True)
