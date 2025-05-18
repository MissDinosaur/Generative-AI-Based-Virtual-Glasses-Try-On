import cv2
import numpy as np
from .face_detection import get_eye_centers

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
