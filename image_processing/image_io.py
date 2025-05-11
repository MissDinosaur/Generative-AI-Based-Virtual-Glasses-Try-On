# Load/save images
import cv2
import requests
from PIL import Image
from io import BytesIO

def load_selfie(path):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_glasses_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGBA")
