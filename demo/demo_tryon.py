import image_processing.image_io as imgio
from image_processing.face_detection import detect_eyes
import image_processing.overlay as ol
from image_processing.frame_utils import remove_white_background
import cv2
import path_utils
import numpy as np


# Step 1: Load the selfie and glasses image
test_selfie = "CM200.jpg"
test_selfie_path = path_utils.get_demo_selfies_path(test_selfie) 
print(f"test_selfie_path: {test_selfie_path}")
test_glasses_url = 'https://optiker-bode.de/sites/default/files/2022-10/0RX4378V__8172_000A.png'


def run_demo_1():
    """Simply ovelay frame onto selfie"""
    selfie_rgb = imgio.load_selfie(test_selfie_path)
    glasses_img = imgio.load_galsses_from_url(test_glasses_url)
    # Step 2: Detect face landmarks
    eyes = detect_eyes(selfie_rgb)

    # Step 3: Resize glasses and overlay glasses on the face
    tryon_result_img = "test_result_1.jpg"
    tryon_result_path = path_utils.get_demo_output_path(tryon_result_img)

    if eyes:
        result = ol.overlay_glasses(selfie_rgb, glasses_img, *eyes)
        print(f"tryon_result_img: {tryon_result_path}")
        # Step 4: Save the try-on result image
        cv2.imwrite(tryon_result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Successfully saved to {tryon_result_path}")
    else:
        print("Face not detected.")

def run_demo_2():
    """The improvement on run_demo_1"""
    # Load images
    face_img = imgio.load_image_from_path(test_selfie_path)
    glasses_img = imgio.load_image_from_url(test_glasses_url)

    # Fuse glasses onto face
    fused_image = ol.fuse_glasses_to_face(face_img, glasses_img)

    # Save result
    tryon_result_img = "test_result_2.jpg"
    tryon_result_path = path_utils.get_demo_output_path(tryon_result_img)
    imgio.save_image(tryon_result_path, fused_image)
    print(f"Fusion complete. Output saved to: {tryon_result_path}")


def run_demo_3():
    """Try to make the lens transparent before overlaying frame to selfie"""
    # Load images
    face_img = imgio.load_selfie(test_selfie_path)
    glasses_img = imgio.load_galsses_from_url(test_glasses_url)
    glasses_img_transparent = remove_white_background(glasses_img)

    # Fuse glasses onto face
    fused_image = ol.overlay_glasses_on_face(face_img, glasses_img_transparent)

    # Save result
    tryon_result_img = "test_result_3.jpg"
    tryon_result_path = path_utils.get_demo_output_path(tryon_result_img)
    imgio.save_image_to_path(tryon_result_path, fused_image)
    print(f"Fusion complete. Output saved to: {tryon_result_path}")


if __name__ == "__main__":
    run_demo_3()