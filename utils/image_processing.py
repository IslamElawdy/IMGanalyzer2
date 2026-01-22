import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from io import BytesIO
import streamlit as st

def load_image(image_file, max_size=1024):
    """Loads an image from a file object, converts to RGB, and resizes if needed."""
    try:
        image = Image.open(image_file).convert('RGB')

        # Resize if too large, maintaining aspect ratio
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def normalize_image(image_np, mode='0-1'):
    """
    Normalizes a numpy image (H, W, C) or (H, W).
    Modes:
    - '0-1': Divides by 255.0
    - '-1-1': Scales to [-1, 1]
    - 'standard': Standardizes (mean=0, std=1) based on image statistics
    """
    img = image_np.astype(np.float32)

    if mode == '0-1':
        return img / 255.0
    elif mode == '-1-1':
        return (img / 127.5) - 1.0
    elif mode == 'standard':
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-7)
    return img

def to_tensor(image_pil):
    """Converts PIL image to PyTorch tensor (C, H, W) in [0, 1]."""
    return transforms.ToTensor()(image_pil)

def tensor_to_display(tensor):
    """Converts a tensor (C, H, W) back to numpy (H, W, C) for display.
       Handles un-normalization if needed, clamps to [0, 1].
    """
    if tensor.ndim == 4: # batch dim
        tensor = tensor[0]

    img = tensor.permute(1, 2, 0).detach().cpu().numpy()

    # Simple Min-Max scaling to visualization range [0, 1] if outside
    if img.min() < 0 or img.max() > 1:
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)

    return np.clip(img, 0, 1)

def show_download_button(image, filename="image.png"):
    """Displays a download button for a PIL Image or Numpy array."""
    if isinstance(image, np.ndarray):
        # Scale if float [0,1]
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        image = Image.fromarray(image)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    st.download_button(
        label="Download Image",
        data=buffered.getvalue(),
        file_name=filename,
        mime="image/png"
    )
