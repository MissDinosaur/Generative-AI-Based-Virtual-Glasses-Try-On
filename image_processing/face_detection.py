import mediapipe as mp
import cv2

def detect_eyes(image_rgb):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None
        
        # Get landmarks
        landmarks = results.multi_face_landmarks[0]
        # Get landmark positions for left & right eye corners
        h, w, _ = image_rgb.shape
        left_eye = landmarks.landmark[33]    # left eye outer
        right_eye = landmarks.landmark[263]  # right eye outer
        return (int(left_eye.x * w), int(left_eye.y * h)), (int(right_eye.x * w), int(right_eye.y * h))
