from image_processing.image_io import load_selfie, load_glasses_from_url
from image_processing.face_detection import detect_eyes
from image_processing.overlay import overlay_glasses
import cv2
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"base_dir: {base_dir}")

# Step 1: Load the selfie
test_selfie = "CM200.jpg"
selfie_path = os.path.join(base_dir, "data", "raw", "selfies")
test_selfie_path = os.path.join(selfie_path, test_selfie)
print(f"test_selfie_path: {test_selfie_path}")
test_glasses_url = 'https://optiker-bode.de/sites/default/files/2022-10/0RX4378V__8172_000A.png'

selfie_rgb = load_selfie(test_selfie_path)
glasses_img = load_glasses_from_url(test_glasses_url)

# Step 2: Detect face landmarks
eyes = detect_eyes(selfie_rgb)

# Step 3: Resize glasses and overlay glasses on the face
tryon_result_path = os.path.join(base_dir, "data", "output")
tryon_result_img = "test_result_1.jpg"
if eyes:
    result = overlay_glasses(selfie_rgb, glasses_img, *eyes) 
    print(f"tryon_result_img: {os.path.join(tryon_result_path, tryon_result_img)}")
    cv2.imwrite(os.path.join(tryon_result_path, tryon_result_img), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"Successfully saved to {os.path.join(tryon_result_path, tryon_result_img)}")
else:
    print("Face not detected.")
