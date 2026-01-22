import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.utils import draw_bounding_boxes
import numpy as np
from PIL import Image
from utils.localization import get_text
import torch.nn.functional as F

@st.cache_resource
def load_detection_model():
    # Faster R-CNN with ResNet-50 FPN backbone
    model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    return model

def render(lang):
    st.header(get_text('module_detection', lang))

    if st.session_state.get('uploaded_image') is None:
        st.warning(get_text('no_image_warning', lang))
        return

    image = st.session_state['uploaded_image']
    st.write(get_text('detect_intro', lang))
    st.info(get_text('math_detection', lang))
    st.latex(r"\text{Loss} = \mathcal{L}_{cls} + \mathcal{L}_{box}")

    try:
        model = load_detection_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    score_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, step=0.05)

    # Preprocess
    # FasterRCNN expects [0, 1] tensors
    t = transforms.ToTensor()
    img_tensor = t(image)

    with torch.no_grad():
        predictions = model([img_tensor])[0]

    # Filter by score
    scores = predictions['scores']
    mask = scores > score_threshold

    boxes = predictions['boxes'][mask]
    labels = predictions['labels'][mask]
    scores = scores[mask]

    # Draw
    # We need COCO labels. torchvision doesn't export them easily in a list,
    # but we can grab a standard list or just show IDs if list not available.
    # Usually FasterRCNN is trained on COCO (91 classes).

    coco_names = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    ]

    pred_labels = [f"{coco_names[i]} {s:.2f}" for i, s in zip(labels, scores)]

    # draw_bounding_boxes expects uint8 tensor [0, 255]
    img_uint8 = (img_tensor * 255).to(torch.uint8)

    if len(boxes) > 0:
        result = draw_bounding_boxes(img_uint8, boxes, labels=pred_labels, width=3, colors="red")
        st.image(result.permute(1, 2, 0).numpy(), caption="Detections", use_container_width=True)
        st.success(f"Detected {len(boxes)} objects.")
    else:
        st.image(image, caption="No objects detected above threshold.", use_container_width=True)
