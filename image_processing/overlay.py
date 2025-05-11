import numpy as np

def overlay_glasses(selfie_rgb, glasses_img, left_eye, right_eye):
    # Resize glasses
    eye_width = int(np.linalg.norm([right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]]) * 2)
    aspect_ratio = glasses_img.height / glasses_img.width
    glasses_resized = glasses_img.resize((eye_width, int(eye_width * aspect_ratio)))
    
    # Overlay glasses on the face
    gx, gy = left_eye[0] - eye_width // 4, left_eye[1] - glasses_resized.height // 2
    glasses_np = np.array(glasses_resized)

    # Overlay (with alpha) alpha?
    for i in range(glasses_np.shape[0]):
        for j in range(glasses_np.shape[1]):
            if gy + i >= selfie_rgb.shape[0] or gx + j >= selfie_rgb.shape[1]:
                continue
            alpha = glasses_np[i, j, 3] / 255.0
            if alpha > 0:
                selfie_rgb[gy + i, gx + j] = (1 - alpha) * selfie_rgb[gy + i, gx + j] + alpha * glasses_np[i, j, :3]
    return selfie_rgb
