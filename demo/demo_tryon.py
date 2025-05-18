from image_processing.image_io import load_selfie, load_glasses_from_url, load_image, load_image_from_url, save_image
from image_processing.face_detection import detect_eyes
from image_processing.overlay import overlay_glasses, fuse_glasses_to_face
import cv2
import path_utils

"""base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"base_dir: {base_dir}")"""

# Step 1: Load the selfie and glasses image
test_selfie = "CM200.jpg"
test_selfie_path = path_utils.get_demo_selfies_path(test_selfie) 
print(f"test_selfie_path: {test_selfie_path}")
test_glasses_url = 'https://optiker-bode.de/sites/default/files/2022-10/0RX4378V__8172_000A.png'


def run_demo_1():
    selfie_rgb = load_selfie(test_selfie_path)
    glasses_img = load_glasses_from_url(test_glasses_url)
    # Step 2: Detect face landmarks
    eyes = detect_eyes(selfie_rgb)

    # Step 3: Resize glasses and overlay glasses on the face
    tryon_result_img = "test_result_1.jpg"
    tryon_result_path = path_utils.get_demo_output_path(tryon_result_img)

    if eyes:
        result = overlay_glasses(selfie_rgb, glasses_img, *eyes)
        print(f"tryon_result_img: {tryon_result_path}")
        # Step 4: Save the try-on result image
        cv2.imwrite(tryon_result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"Successfully saved to {tryon_result_path}")
    else:
        print("Face not detected.")

def run_demo_2():
    # Load images
    face_img = load_image(test_selfie_path)
    glasses_img = load_image_from_url(test_glasses_url)

    # Fuse glasses onto face
    fused_image = fuse_glasses_to_face(face_img, glasses_img)

    # Save result
    tryon_result_img = "test_result_2.jpg"
    tryon_result_path = path_utils.get_demo_output_path(tryon_result_img)
    save_image(tryon_result_path, fused_image)
    print(f"✅ Fusion complete. Output saved to: {tryon_result_path}")


if __name__ == "__main__":
    run_demo_2()