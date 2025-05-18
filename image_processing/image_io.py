# Load/save images
import cv2
import requests
from PIL import Image
from io import BytesIO
import numpy as np

def load_selfie(path):
    """Load an image from the given path."""
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_glasses_from_url(url):
    """Load an image from the given path."""
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGBA")

def load_image_from_url(url):
    """Load an image from a URL."""
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download image from URL: {url}")
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def load_image(path):
    """Load an image from the given path."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return image

def save_image(path, image):
    """Save an image to the given path."""
    cv2.imwrite(path, image)