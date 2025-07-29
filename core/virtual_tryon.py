"""
Virtual try-on core functionality.
"""
import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import tempfile
import os
import logging
from core.image_utils import image_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_keypoints(face_img: np.ndarray) -> dict:
    """Extract facial landmarks using MediaPipe - improved version."""
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)
    
    h, w = face_img.shape[:2]
    rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_img)
    
    if not results.multi_face_landmarks:
        raise ValueError("No face detected")
    
    landmarks = results.multi_face_landmarks[0]
    
    def get_point(idx):
        point = landmarks.landmark[idx]
        return np.array([point.x * w, point.y * h])
    
    # Eye centers
    L_eye = (get_point(33) + get_point(133)) / 2
    R_eye = (get_point(362) + get_point(263)) / 2
    
    # Nose points
    nose_tip = get_point(1)
    nose_root = get_point(168)
    nose_center = (nose_tip + nose_root) / 2
    
    # Ear points
    L_ear = get_point(234)
    R_ear = get_point(454)
    
    # Eye and nose line vectors
    eye_line_vec = R_eye - L_eye
    nose_vec = nose_tip - nose_root
    
    # Face center using line intersection
    C1 = line_intersection(L_eye, eye_line_vec, nose_center, nose_vec)
    
    # Calculate ear distance
    ear_distance = np.linalg.norm(R_ear - L_ear)
    
    # Face angle
    face_angle = np.degrees(np.arctan2(R_eye[1] - L_eye[1], R_eye[0] - L_eye[0]))
    
    return {
        'L_eye': L_eye,
        'R_eye': R_eye,
        'L_ear': L_ear,
        'R_ear': R_ear,
        'C1': C1,
        'ear_distance': ear_distance,
        'face_angle': face_angle
    }

def compute_glasses_center(img_rgba: np.ndarray) -> np.ndarray:
    """Compute glasses center - improved version."""
    alpha = img_rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    
    if len(xs) == 0:
        h, w = img_rgba.shape[:2]
        return np.array([w/2, h/2])
    
    top, bottom = ys.min(), ys.max()
    left, right = xs.min(), xs.max()
    
    # Focus on top 40% for lens centers
    y_threshold = top + (bottom - top) * 0.4
    center_x = (left + right) / 2
    
    # Left lens
    is_left = (xs < center_x) & (ys < y_threshold)
    if np.any(is_left):
        pts_l_top = np.vstack([xs[is_left], ys[is_left]]).T
        C_left = pts_l_top.mean(axis=0)
    else:
        C_left = np.array([left + (right-left)*0.25, top + (bottom-top)*0.3])
    
    # Right lens
    is_right = (xs >= center_x) & (ys < y_threshold)
    if np.any(is_right):
        pts_r_top = np.vstack([xs[is_right], ys[is_right]]).T
        C_right = pts_r_top.mean(axis=0)
    else:
        C_right = np.array([right - (right-left)*0.25, top + (bottom-top)*0.3])
    
    return (C_left + C_right) / 2

def remove_background_simple(img_bgr: np.ndarray) -> np.ndarray:
    """Simple background removal - ONLY for pure black/white backgrounds."""
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Only remove PURE white and PURE black pixels
    pure_white_mask = gray > 250  # Very high threshold
    pure_black_mask = gray < 5    # Very low threshold
    background_mask = pure_white_mask | pure_black_mask
    
    # Create alpha channel
    alpha = np.where(background_mask, 0, 255).astype(np.uint8)
    
    # Convert to RGBA
    img_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    img_rgba[:, :, 3] = alpha
    
    return img_rgba

def detect_and_remove_arms_advanced(img_rgba: np.ndarray) -> np.ndarray:
    """Advanced arm detection using multiple methods."""
    if img_rgba.shape[2] != 4:
        return img_rgba
    
    h, w = img_rgba.shape[:2]
    result = img_rgba.copy()
    
    # Get alpha channel for structure analysis
    alpha = img_rgba[:, :, 3]
    
    # Method 1: Edge-based detection
    # Find edges in alpha channel
    edges = cv2.Canny(alpha, 50, 150)
    
    # Find contours from edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    arm_mask = np.zeros_like(alpha)
    
    for contour in contours:
        # Get bounding rectangle
        x, y, cw, ch = cv2.boundingRect(contour)
        
        # Check if it's in arm regions (outer 25% of width)
        if x < w * 0.25 or x + cw > w * 0.75:
            # Check aspect ratio - arms are typically elongated
            aspect_ratio = cw / max(ch, 1)
            
            # If it's horizontal and in arm region, likely an arm
            if aspect_ratio > 2.0 or ch < h * 0.1:
                cv2.fillPoly(arm_mask, [contour], 255)
    
    # Method 2: Morphological arm detection
    # Create horizontal kernel to detect arm-like structures
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    horizontal_structures = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_h)
    
    # Only keep structures in arm regions
    horizontal_structures[:, int(w * 0.25):int(w * 0.75)] = 0
    
    # Combine masks
    combined_mask = cv2.bitwise_or(arm_mask, horizontal_structures)
    
    # Clean up the mask
    kernel_clean = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_clean)
    
    # Apply removal
    result[combined_mask > 0] = [0, 0, 0, 0]
    
    return result

def remove_lens_region(img_rgba: np.ndarray) -> np.ndarray:
    """Remove lens regions - EXTREMELY CONSERVATIVE."""
    if img_rgba.shape[2] != 4:
        return img_rgba
    
    # Skip lens removal entirely for now - too risky
    return img_rgba

def smooth_frame_edge(img_rgba: np.ndarray, 
                     white_thresh: int = 254,
                     alpha_thresh: int = 30,
                     kernel_size: int = 1) -> np.ndarray:
    """Minimal edge cleaning - ONLY pure white fringes."""
    if img_rgba.shape[2] != 4:
        return img_rgba
    
    result = img_rgba.copy()
    rgb = result[:, :, :3]
    alpha = result[:, :, 3]
    
    # Only remove PURE white pixels with very low alpha
    fringe_mask = np.logical_and(
        alpha < alpha_thresh,
        np.all(rgb > white_thresh, axis=2)
    ).astype(np.uint8) * 255
    
    result[fringe_mask == 255] = [0, 0, 0, 0]
    
    return result

def line_intersection(point_1, line_vec_1, point_2, line_vec_2):
    """Compute intersection of two lines."""
    A = np.array([line_vec_1, -line_vec_2]).T
    if np.linalg.matrix_rank(A) < 2:
        return (point_1 + point_2) / 2
    b = point_2 - point_1
    t_s = np.linalg.solve(A, b)
    return point_1 + t_s[0] * line_vec_1

def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image around center."""
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    if img.shape[2] == 4:  # RGBA
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    else:  # RGB
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    
    return rotated

def rotate_point(point: np.ndarray, center: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a point around center by angle (degrees)."""
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Translate to origin
    translated = point - center
    
    # Rotate
    rotated = np.array([
        translated[0] * cos_a - translated[1] * sin_a,
        translated[0] * sin_a + translated[1] * cos_a
    ])
    
    # Translate back
    return rotated + center

def overlay_glasses(glasses: np.ndarray, face: np.ndarray, x: int, y: int) -> np.ndarray:
    """Overlay glasses on face using improved alpha blending."""
    gh, gw = glasses.shape[:2]
    fh, fw = face.shape[:2]
    
    # Calculate valid regions
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(fw, x + gw)
    y2 = min(fh, y + gh)
    
    # Glasses region
    gx1 = x1 - x
    gy1 = y1 - y
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)
    
    if x2 <= x1 or y2 <= y1:
        return face
    
    # Extract regions
    face_region = face[y1:y2, x1:x2].astype(np.float32)
    glasses_region = glasses[gy1:gy2, gx1:gx2].astype(np.float32)
    
    # Check if regions have valid shapes
    if face_region.shape[:2] != glasses_region.shape[:2]:
        return face
    
    # Enhanced alpha blending with edge smoothing
    alpha = glasses_region[:, :, 3:4] / 255.0
    
    # Apply Gaussian blur to alpha for smoother edges
    alpha_blurred = cv2.GaussianBlur(alpha, (3, 3), 0.5)
    
    # Ensure alpha has correct shape for broadcasting
    if len(alpha_blurred.shape) == 3:
        alpha_blurred = alpha_blurred[:, :, 0:1]
    else:
        alpha_blurred = alpha_blurred[:, :, np.newaxis]
    
    # Color correction for better realism
    glasses_rgb = glasses_region[:, :, :3]
    
    # Slight color adjustment to match face lighting
    try:
        face_mean = np.mean(face_region, axis=(0, 1))
        valid_mask = alpha[:,:,0] > 0.1
        if np.any(valid_mask):
            glasses_mean = np.mean(glasses_rgb[valid_mask], axis=0)
            
            if len(glasses_mean) == 3 and np.all(glasses_mean > 1e-6):
                color_ratio = face_mean / glasses_mean
                color_ratio = np.clip(color_ratio, 0.8, 1.2)  # Limit adjustment
                glasses_rgb = glasses_rgb * color_ratio[np.newaxis, np.newaxis, :]
                glasses_rgb = np.clip(glasses_rgb, 0, 255)
    except:
        pass  # Skip color correction if it fails
    
    # Final blending
    result = face.copy().astype(np.float32)
    blended = (alpha_blurred * glasses_rgb + (1 - alpha_blurred) * face_region)
    result[y1:y2, x1:x2] = np.clip(blended, 0, 255)
    
    return result.astype(np.uint8)

def remove_glasses_arms(img_rgba: np.ndarray) -> np.ndarray:
    """Remove glasses arms using HSV filtering - SIMPLE approach."""
    if img_rgba.shape[2] != 4:
        return img_rgba
    
    # Convert to HSV
    rgb = img_rgba[:, :, :3]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    
    # HSV range for arms: bright, low saturation areas
    lower = np.array([0, 0, 200])    # H: 0-180, S: 0-30, V: 200-255
    upper = np.array([180, 30, 255])
    
    # Create mask for arm pixels
    arm_mask = cv2.inRange(hsv, lower, upper)
    
    # Only process visible pixels (alpha > 0)
    alpha_mask = img_rgba[:, :, 3] > 0
    arm_mask = cv2.bitwise_and(arm_mask, arm_mask, mask=alpha_mask.astype(np.uint8) * 255)
    
    # Find contours and apply convex hull
    contours, _ = cv2.findContours(arm_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = img_rgba.copy()
    
    for contour in contours:
        # Apply convex hull to smooth the region
        hull = cv2.convexHull(contour)
        # Set alpha = 0 (fully transparent)
        cv2.fillPoly(result, [hull], [0, 0, 0, 0])
    
    removed_pixels = np.sum(img_rgba[:,:,3] > 0) - np.sum(result[:,:,3] > 0)
    logger.info(f"Arm removal: removed {removed_pixels} pixels")
    
    return result

def remove_glasses_arms_conservative(img_rgba: np.ndarray) -> np.ndarray:
    """Remove glasses arms - ULTRA CONSERVATIVE approach."""
    if img_rgba.shape[2] != 4:
        return img_rgba
    
    h, w = img_rgba.shape[:2]
    result = img_rgba.copy()
    
    # Only target the extreme edges where arms extend beyond the face
    # Arms typically extend in the outer 15% of the image
    left_arm_zone = slice(0, int(w * 0.15))
    right_arm_zone = slice(int(w * 0.85), w)
    
    # Method 1: Remove only very thin horizontal structures in extreme edges
    alpha = img_rgba[:, :, 3]
    
    # Create horizontal kernel to detect very thin arm-like structures
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 2))
    thin_structures = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel_thin)
    
    # Only keep structures in extreme edge zones
    arm_mask = np.zeros_like(thin_structures)
    arm_mask[:, left_arm_zone] = thin_structures[:, left_arm_zone]
    arm_mask[:, right_arm_zone] = thin_structures[:, right_arm_zone]
    
    # Method 2: Remove isolated pixels in arm zones
    # Find connected components in arm zones
    for zone in [left_arm_zone, right_arm_zone]:
        zone_alpha = alpha[:, zone].copy()
        
        # Find small isolated components
        num_labels, labels = cv2.connectedComponents(zone_alpha)
        
        for label in range(1, num_labels):
            component_mask = (labels == label)
            component_size = np.sum(component_mask)
            
            # Remove very small isolated components (likely arm fragments)
            if component_size < 200:  # Very small threshold
                zone_start = zone.start if zone.start else 0
                zone_coords = np.where(component_mask)
                result[zone_coords[0], zone_coords[1] + zone_start] = [0, 0, 0, 0]
    
    # Apply the thin structure removal
    result[arm_mask > 0] = [0, 0, 0, 0]
    
    removed_pixels = np.sum(alpha > 0) - np.sum(result[:,:,3] > 0)
    logger.info(f"Conservative arm removal: removed {removed_pixels} pixels")
    
    return result

def main_tryon_from_binary(selfie_binary_data, glasses_url, 
                          gl_base_scale_param: float = 1.5,
                          gl_upward_pixels: int = 10,
                          show_result: bool = False,
                          save_result: str = None):
    """Main virtual try-on function - PRESERVE GLASSES AT ALL COSTS."""
    temp_selfie_path = None
    
    try:
        # Save selfie to temp file
        temp_selfie_path = image_utils.save_binary_to_temp(selfie_binary_data)
        
        # Load face image
        face = cv2.imread(temp_selfie_path)
        if face is None:
            raise ValueError("Failed to load selfie")
        
        # Load glasses from URL
        resp = urllib.request.urlopen(glasses_url)
        img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
        glasses = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        
        if glasses is None:
            raise ValueError("Failed to load glasses")
        
        # Convert to RGBA if needed
        if glasses.shape[2] == 3:
            glasses_rgba = remove_background_simple(glasses)
        else:
            glasses_rgba = glasses
        
        logger.info(f"Face: {face.shape}, Glasses: {glasses_rgba.shape}")
        logger.info(f"Initial glasses pixels: {np.sum(glasses_rgba[:,:,3] > 0)}")
        
        # STEP 1: Clean pure white fringes
        glasses_rgba = smooth_frame_edge(glasses_rgba, white_thresh=254, alpha_thresh=30, kernel_size=1)
        logger.info(f"After edge cleaning: {np.sum(glasses_rgba[:,:,3] > 0)} pixels")
        
        # STEP 2: Remove glasses arms
        glasses_rgba = remove_glasses_arms(glasses_rgba)
        logger.info(f"After arm removal: {np.sum(glasses_rgba[:,:,3] > 0)} pixels")
        
        # Get face keypoints
        face_kp = get_keypoints(face)
        C1 = face_kp['C1']
        ear_distance = face_kp['ear_distance']
        face_angle = face_kp['face_angle']
        
        # Get glasses center
        C2 = compute_glasses_center(glasses_rgba)
        
        # Calculate scale and resize
        scale = (ear_distance * gl_base_scale_param) / glasses_rgba.shape[1]
        glasses_scaled = cv2.resize(glasses_rgba, (0, 0), fx=scale, fy=scale)
        
        # Rotate glasses
        glasses_rotated = rotate_image(glasses_scaled, -face_angle)
        
        # Update center after transformations
        C2_scaled = C2 * scale
        h_rot, w_rot = glasses_rotated.shape[:2]
        center_rot = np.array([w_rot/2, h_rot/2])
        C2_final = rotate_point(C2_scaled, center_rot, -face_angle)
        
        # Position glasses
        top_left = C1 - C2_final
        top_left[1] -= gl_upward_pixels
        x, y = int(top_left[0]), int(top_left[1])
        
        # Final check before overlay
        final_pixels = np.sum(glasses_rotated[:,:,3] > 0)
        logger.info(f"Final glasses pixels before overlay: {final_pixels}")
        
        # Overlay glasses
        result = overlay_glasses(glasses_rotated, face, x, y)
        
        if save_result:
            cv2.imwrite(save_result, result)
            logger.info(f"Result saved to: {save_result}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in virtual try-on: {e}")
        raise
    
    finally:
        if temp_selfie_path:
            image_utils.cleanup_temp_file(temp_selfie_path)

def main_tryon(selfie_path: str, glasses_url: str, 
               gl_base_scale_param: float = 1.2, 
               gl_upward_pixels: int = 5):
    """Backward compatibility function."""
    with open(selfie_path, 'rb') as f:
        selfie_binary = f.read()
    
    return main_tryon_from_binary(selfie_binary, glasses_url, 
                                 gl_base_scale_param, gl_upward_pixels)
