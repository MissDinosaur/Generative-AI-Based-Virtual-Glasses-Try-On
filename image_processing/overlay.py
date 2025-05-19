import cv2
from PIL import Image
import numpy as np
from .face_detection import get_eye_centers, get_eye_centers_2

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

def align_glasses(glasses, eye_pts_src, eye_pts_dst):
    d_src = eye_pts_src[1] - eye_pts_src[0]
    d_dst = eye_pts_dst[1] - eye_pts_dst[0]
    angle = np.arctan2(d_dst[1], d_dst[0]) - np.arctan2(d_src[1], d_src[0])
    scale = np.linalg.norm(d_dst) / np.linalg.norm(d_src)

    center = tuple(eye_pts_src[0])
    rot_mat = cv2.getRotationMatrix2D(center, np.degrees(angle), scale)

    dx, dy = eye_pts_dst[0] - eye_pts_src[0]
    rot_mat[0, 2] += dx
    rot_mat[1, 2] += dy

    h, w = glasses.shape[:2]
    transformed = cv2.warpAffine(glasses, rot_mat, (w, h), flags=cv2.INTER_LINEAR)
    return transformed


def fuse_glasses_to_face(face_img, glasses_img):
    eyes = get_eye_centers(face_img)
    if eyes is None:
        raise ValueError("No face detected in the image.")

    left_eye, right_eye = eyes

    h, w = glasses_img.shape[:2]
    g_left = np.array([w * 1/3, h // 2])
    g_right = np.array([w * 2/3, h // 2])

    aligned_glasses = align_glasses(glasses_img, [g_left, g_right], [left_eye, right_eye])

    gray = cv2.cvtColor(aligned_glasses, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    center = tuple(((left_eye + right_eye) // 2).astype(int))
    fused = cv2.seamlessClone(aligned_glasses, face_img, mask, center, cv2.NORMAL_CLONE)

    return fused


def resize_glasses_to_face(glasses_rgba_pil: Image.Image, left_eye, right_eye, scale_factor=1.8) -> Image.Image:
    """
    Resize the glasses image based on eye distance and a scale factor.
    """
    eye_distance = np.linalg.norm(np.array(right_eye) - np.array(left_eye))
    target_width = int(eye_distance * scale_factor)

    original_width, original_height = glasses_rgba_pil.size
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    resized = glasses_rgba_pil.resize((target_width, target_height), Image.LANCZOS)
    return resized


def overlay_glasses_on_face(face_rgb: np.ndarray, glasses_rgba_pil: Image.Image) -> np.ndarray:
    """
    Align and overlay RGBA glasses image onto the RGB face image.
    """
    left_eye, right_eye = get_eye_centers_2(face_rgb)

    # Resize glasses based on eye distance
    glasses_rgba_pil = resize_glasses_to_face(glasses_rgba_pil, left_eye, right_eye)

    glasses_rgba = np.array(glasses_rgba_pil)
    gh, gw = glasses_rgba.shape[:2]

    # Define glasses' left and right lens center in glasses image (as fractions of width)
    glasses_left_center_x = int(gw * 0.3)   # ~30% from left
    glasses_right_center_x = int(gw * 0.7)  # ~70% from left
    glasses_center_y = gh // 2               # vertical center

    # Calculate top-left corner to place the glasses so that
    # glasses_left_center aligns with left_eye,
    # glasses_right_center aligns with right_eye.
    # Compute the offset by averaging the required position:
    offset_x = int((left_eye[0] - glasses_left_center_x + right_eye[0] - glasses_right_center_x) / 2)
    offset_y = int(left_eye[1] - glasses_center_y)  # Assume vertical alignment with left eye

    face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    alpha = glasses_rgba[:, :, 3] / 255.0

    for y in range(gh):
        for x in range(gw):
            fx = offset_x + x
            fy = offset_y + y
            if 0 <= fx < face_bgr.shape[1] and 0 <= fy < face_bgr.shape[0]:
                for c in range(3):
                    face_bgr[fy, fx, c] = (
                        alpha[y, x] * glasses_rgba[y, x, c] +
                        (1 - alpha[y, x]) * face_bgr[fy, fx, c]
                    )

    face_out = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return face_out
