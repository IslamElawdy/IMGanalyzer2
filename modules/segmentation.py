import streamlit as st
import torch
from torchvision import models, transforms
import numpy as np
from PIL import Image
from utils.localization import get_text
from torchvision.utils import draw_segmentation_masks

@st.cache_resource
def load_semantic_model():
    model = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
    model.eval()
    return model

@st.cache_resource
def load_instance_model():
    model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model

def render(lang):
    st.header(get_text('module_segmentation', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    st.write(get_text('segmentation_intro', lang))

    tab_sem, tab_inst = st.tabs(["Semantic Segmentation (DeepLabV3)", "Instance Segmentation (Mask R-CNN)"])

    # -------------------------------------------------------------------------
    # Semantic Segmentation
    # -------------------------------------------------------------------------
    with tab_sem:
        st.subheader("Semantic Segmentation")
        st.write("Ordnet jedem Pixel eine Klasse zu. Unterscheidet NICHT zwischen verschiedenen Objekten derselben Klasse (z.B. zwei Autos sind einfach 'Auto-Pixel').")
        st.info("Klassen (COCO Subset): Background, Aeroplane, Bicycle, Bird, Boat, Bottle, Bus, Car, Cat, Chair, Cow, DiningTable, Dog, Horse, Motorbike, Person, PottedPlant, Sheep, Sofa, Train, TV/Monitor")

        if st.button("Run Semantic Segmentation"):
            try:
                model = load_semantic_model()
                preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224), # Cropping for display consistency
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                # To show overlay correctly, we need the cropped original
                w, h = image.size
                # Logic to mimic CenterCrop(224) on PIL
                left = (w - 224)/2
                top = (h - 224)/2
                right = (w + 224)/2
                bottom = (h + 224)/2
                image_crop = image.crop((left, top, right, bottom))

                input_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)['out'][0]

                output_predictions = output.argmax(0)

                # Color Palette
                palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                colors = (colors % 255).numpy().astype("uint8")

                r = Image.fromarray(output_predictions.byte().cpu().numpy())
                r.putpalette(colors)

                c1, c2 = st.columns(2)
                c1.image(image_crop, caption="Input (Center Crop)", use_container_width=True)
                c2.image(r, caption="Prediction Mask", use_container_width=True)

                # Overlay
                mask_rgb = r.convert("RGB")
                overlay = Image.blend(image_crop, mask_rgb, 0.5)
                st.image(overlay, caption="Overlay", use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")

    # -------------------------------------------------------------------------
    # Instance Segmentation
    # -------------------------------------------------------------------------
    with tab_inst:
        st.subheader("Instance Segmentation")
        st.write("Erkennt und maskiert einzelne Objekte. Unterscheidet zwischen 'Auto 1' und 'Auto 2'.")

        score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)

        if st.button("Run Instance Segmentation"):
            try:
                model = load_instance_model()
                # Mask R-CNN expects 0-1 tensor
                t_inst = transforms.ToTensor()
                img_tensor = t_inst(image)

                with torch.no_grad():
                    prediction = model([img_tensor])[0]

                # Filter by score
                scores = prediction['scores']
                mask_filter = scores > score_threshold

                masks = prediction['masks'][mask_filter]
                labels = prediction['labels'][mask_filter]

                if len(masks) > 0:
                    st.success(f"Detected {len(masks)} objects.")

                    # Masks are (N, 1, H, W) -> squeeze to (N, H, W)
                    masks = masks.squeeze(1)
                    # Boolean mask
                    masks = masks > 0.5

                    # Draw
                    img_uint8 = (img_tensor * 255).to(torch.uint8)
                    res_tensor = draw_segmentation_masks(img_uint8, masks, alpha=0.5)
                    res_pil = transforms.ToPILImage()(res_tensor)

                    st.image(res_pil, caption="Instance Segmentation Result", use_container_width=True)
                else:
                    st.warning("No objects detected above threshold.")

            except Exception as e:
                st.error(f"Error: {e}")
