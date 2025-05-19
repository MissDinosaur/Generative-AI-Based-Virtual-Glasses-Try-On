import mediapipe as mp
import cv2
import numpy as np

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

def get_eye_centers(image):
    """Return (left_eye, right_eye) center points using MediaPipe FaceMesh."""
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        LEFT_EYE = [33, 133]
        RIGHT_EYE = [362, 263]
        h, w = image.shape[:2]

        left_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE], axis=0)
        right_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE], axis=0)

        return np.array(left_eye), np.array(right_eye)

def get_eye_centers_2(image_rgb):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        ih, iw = image_rgb.shape[:2]
        left_eye_idx = [33, 133]
        right_eye_idx = [362, 263]

        left_eye = np.mean([(landmarks[i].x * iw, landmarks[i].y * ih) for i in left_eye_idx], axis=0)
        right_eye = np.mean([(landmarks[i].x * iw, landmarks[i].y * ih) for i in right_eye_idx], axis=0)

        return np.array(left_eye), np.array(right_eye)