"""
Accuracy evaluation script for virtual try-on system.
Measures various metrics to assess try-on quality.
"""
import cv2
import numpy as np
import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.database_config import db_config
from core.virtual_tryon import main_tryon_from_binary
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualTryOnAccuracyCalculator:
    def __init__(self):
        self.schema = db_config.schema
        self.mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
    def get_test_samples(self, count=50):
        """Get random samples for testing."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Get random selfies
            cursor.execute(f"""
                SELECT id, filename, gender, race, image_data 
                FROM {self.schema}.selfies 
                ORDER BY RANDOM() 
                LIMIT %s
            """, (count,))
            
            selfies = cursor.fetchall()
            
            # Get random glasses
            cursor.execute(f"""
                SELECT id, brand, title, main_image 
                FROM {self.schema}.frames 
                WHERE main_image IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT %s
            """, (count,))
            
            glasses = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return selfies, glasses
            
        except Exception as e:
            logger.error(f"Error getting test samples: {e}")
            return [], []
    
    def detect_face_landmarks(self, image):
        """Detect face landmarks in image."""
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mp_face.process(rgb_image)
            
            if results.multi_face_landmarks:
                return results.multi_face_landmarks[0]
            return None
            
        except Exception as e:
            logger.error(f"Error detecting landmarks: {e}")
            return None
    
    def calculate_alignment_accuracy(self, original_face, result_image):
        """Calculate how well glasses are aligned with face."""
        try:
            # Detect landmarks in both images
            original_landmarks = self.detect_face_landmarks(original_face)
            result_landmarks = self.detect_face_landmarks(result_image)
            
            if not original_landmarks or not result_landmarks:
                return 0.0
            
            h, w = original_face.shape[:2]
            
            # Get eye positions from original
            def get_point(landmarks, idx):
                point = landmarks.landmark[idx]
                return np.array([point.x * w, point.y * h])
            
            # Original eye positions
            orig_left_eye = (get_point(original_landmarks, 33) + get_point(original_landmarks, 133)) / 2
            orig_right_eye = (get_point(original_landmarks, 362) + get_point(original_landmarks, 263)) / 2
            
            # Result eye positions
            result_left_eye = (get_point(result_landmarks, 33) + get_point(result_landmarks, 133)) / 2
            result_right_eye = (get_point(result_landmarks, 362) + get_point(result_landmarks, 263)) / 2
            
            # Calculate eye distance consistency
            orig_eye_distance = np.linalg.norm(orig_right_eye - orig_left_eye)
            result_eye_distance = np.linalg.norm(result_right_eye - result_left_eye)
            
            # Calculate alignment score (closer to 1.0 is better)
            distance_ratio = min(orig_eye_distance, result_eye_distance) / max(orig_eye_distance, result_eye_distance)
            
            # Calculate position accuracy
            left_eye_shift = np.linalg.norm(orig_left_eye - result_left_eye)
            right_eye_shift = np.linalg.norm(orig_right_eye - result_right_eye)
            avg_shift = (left_eye_shift + right_eye_shift) / 2
            
            # Normalize shift by face size
            position_accuracy = max(0, 1 - (avg_shift / orig_eye_distance))
            
            # Combined alignment score
            alignment_score = (distance_ratio + position_accuracy) / 2
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Error calculating alignment: {e}")
            return 0.0
    
    def calculate_realism_score(self, result_image):
        """Calculate realism based on image quality metrics - ENHANCED VERSION."""
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness (Laplacian variance) - IMPROVED
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 500)  # Lower threshold for better scoring
            
            # 2. Contrast (standard deviation) - IMPROVED
            contrast_score = min(1.0, gray.std() / 80)  # Lower threshold
            
            # 3. Brightness consistency - IMPROVED
            mean_brightness = gray.mean()
            brightness_score = 1.0 - abs(mean_brightness - 128) / 128
            
            # 4. Edge preservation - IMPROVED
            edges = cv2.Canny(gray, 30, 100)  # Lower thresholds for more edges
            edge_density = np.sum(edges > 0) / edges.size
            edge_score = min(1.0, edge_density * 15)  # Better normalization
            
            # 5. NEW: Noise level (lower is better)
            # Use bilateral filter to estimate noise
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            noise_level = np.mean(np.abs(gray.astype(float) - filtered.astype(float)))
            noise_score = max(0, 1.0 - noise_level / 20)  # Lower noise = higher score
            
            # 6. NEW: Local contrast (CLAHE response)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            clahe_img = clahe.apply(gray)
            local_contrast = np.std(clahe_img - gray)
            local_contrast_score = min(1.0, local_contrast / 30)
            
            # Combined realism score with better weights
            realism_score = (
                sharpness_score * 0.25 + 
                contrast_score * 0.20 + 
                brightness_score * 0.15 + 
                edge_score * 0.20 + 
                noise_score * 0.10 + 
                local_contrast_score * 0.10
            )
            
            return realism_score
            
        except Exception as e:
            logger.error(f"Error calculating realism: {e}")
            return 0.0
    
    def calculate_glasses_preservation(self, original_glasses_url, result_image):
        """Calculate how well glasses features are preserved - IMPROVED VERSION."""
        try:
            # Load original glasses for comparison
            import urllib.request
            resp = urllib.request.urlopen(original_glasses_url)
            img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            original_glasses = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            
            if original_glasses is None:
                return 0.0
            
            # Convert result to grayscale
            result_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Edge-based detection
            edges = cv2.Canny(result_gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Method 2: Look for glasses-like regions in the eye area
            h, w = result_image.shape[:2]
            eye_region = result_gray[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.8)]
            
            # Find contours in eye region
            contours, _ = cv2.findContours(
                cv2.threshold(eye_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            glasses_features = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 20000:  # Reasonable glasses component size
                    # Check aspect ratio
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = cw / max(ch, 1)
                    if 1.5 < aspect_ratio < 4.0:  # Glasses-like aspect ratio
                        glasses_features += 1
            
            # Method 3: Color consistency check
            # Check if there are non-skin-tone colors in eye region (indicating glasses)
            eye_region_color = result_image[int(h*0.3):int(h*0.6), int(w*0.2):int(w*0.8)]
            hsv_eye = cv2.cvtColor(eye_region_color, cv2.COLOR_BGR2HSV)
            
            # Non-skin colors (glasses frames are usually not skin-colored)
            non_skin_mask = cv2.inRange(hsv_eye, np.array([0, 30, 50]), np.array([20, 255, 255]))
            non_skin_ratio = np.sum(non_skin_mask > 0) / non_skin_mask.size
            
            # Combine all metrics
            edge_score = min(1.0, edge_density * 20)  # Normalize edge density
            feature_score = min(1.0, glasses_features / 3)  # Expect 2-3 features
            color_score = min(1.0, non_skin_ratio * 10)  # Normalize color ratio
            
            preservation_score = (edge_score * 0.4 + feature_score * 0.4 + color_score * 0.2)
            
            return preservation_score
            
        except Exception as e:
            logger.error(f"Error calculating preservation: {e}")
            return 0.0
    
    def run_single_evaluation(self, selfie_data, glasses_data):
        """Run evaluation on a single selfie-glasses pair."""
        try:
            # Run virtual try-on with timeout protection
            result_img = main_tryon_from_binary(
                selfie_data[4],  # image_data
                glasses_data[3]  # main_image URL
            )
            
            # Validate result image
            if result_img is None or result_img.size == 0:
                raise ValueError("Virtual try-on returned invalid result")
            
            # Load original face for comparison
            temp_path = f"/tmp/temp_face_{selfie_data[0]}.jpg"
            with open(temp_path, 'wb') as f:
                f.write(selfie_data[4])
            original_face = cv2.imread(temp_path)
            
            # Clean up temp file immediately
            try:
                os.remove(temp_path)
            except:
                pass
            
            if original_face is None:
                raise ValueError("Failed to load original face")
            
            # Calculate metrics with error handling
            try:
                alignment_score = self.calculate_alignment_accuracy(original_face, result_img)
            except Exception as e:
                logger.warning(f"Alignment calculation failed: {e}")
                alignment_score = 0.0
            
            try:
                realism_score = self.calculate_realism_score(result_img)
            except Exception as e:
                logger.warning(f"Realism calculation failed: {e}")
                realism_score = 0.0
            
            try:
                preservation_score = self.calculate_glasses_preservation(glasses_data[3], result_img)
            except Exception as e:
                logger.warning(f"Preservation calculation failed: {e}")
                preservation_score = 0.0
            
            # Overall accuracy (weighted average)
            overall_accuracy = (alignment_score * 0.4 + realism_score * 0.3 + preservation_score * 0.3)
            
            return {
                'selfie_id': selfie_data[0],
                'selfie_filename': selfie_data[1],
                'glasses_id': glasses_data[0],
                'glasses_brand': glasses_data[1],
                'alignment_score': alignment_score,
                'realism_score': realism_score,
                'preservation_score': preservation_score,
                'overall_accuracy': overall_accuracy,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error in single evaluation: {e}")
            return {
                'selfie_id': selfie_data[0],
                'glasses_id': glasses_data[0],
                'success': False,
                'error': str(e)
            }
    
    def run_batch_evaluation(self, sample_count=50):
        """Run evaluation on multiple samples."""
        logger.info(f"🔍 Starting accuracy evaluation with {sample_count} samples...")
        
        # Get test samples
        selfies, glasses = self.get_test_samples(sample_count)
        
        if not selfies or not glasses:
            logger.error("No test samples available")
            return None
        
        results = []
        successful_runs = 0
        
        for i in range(min(len(selfies), len(glasses), sample_count)):
            logger.info(f"Evaluating sample {i+1}/{sample_count}...")
            
            result = self.run_single_evaluation(selfies[i], glasses[i])
            results.append(result)
            
            if result['success']:
                successful_runs += 1
                logger.info(f"✅ Sample {i+1}: Overall accuracy = {result['overall_accuracy']:.3f}")
            else:
                logger.warning(f"❌ Sample {i+1}: Failed - {result.get('error', 'Unknown error')}")
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            summary = {
                'total_samples': sample_count,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / sample_count,
                'avg_alignment_score': np.mean([r['alignment_score'] for r in successful_results]),
                'avg_realism_score': np.mean([r['realism_score'] for r in successful_results]),
                'avg_preservation_score': np.mean([r['preservation_score'] for r in successful_results]),
                'avg_overall_accuracy': np.mean([r['overall_accuracy'] for r in successful_results]),
                'std_overall_accuracy': np.std([r['overall_accuracy'] for r in successful_results]),
                'min_accuracy': min([r['overall_accuracy'] for r in successful_results]),
                'max_accuracy': max([r['overall_accuracy'] for r in successful_results])
            }
        else:
            summary = {'error': 'No successful evaluations'}
        
        return {
            'timestamp': datetime.now().isoformat(),
            'summary': summary,
            'detailed_results': results
        }
    
    def save_results(self, results, filename=None):
        """Save evaluation results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"accuracy_evaluation_{timestamp}.json"
        
        output_dir = project_root / "evaluation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {filepath}")
        return filepath
    
    def generate_report(self, results):
        """Generate a readable accuracy report."""
        if 'error' in results.get('summary', {}):
            print("❌ Evaluation failed - no successful runs")
            return
        
        summary = results['summary']
        
        print("\n" + "="*60)
        print("🎯 VIRTUAL TRY-ON ACCURACY REPORT")
        print("="*60)
        print(f"📊 Test Summary:")
        print(f"   Total Samples: {summary['total_samples']}")
        print(f"   Successful Runs: {summary['successful_runs']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        
        print(f"\n📈 Accuracy Metrics:")
        print(f"   Overall Accuracy: {summary['avg_overall_accuracy']:.3f} ± {summary['std_overall_accuracy']:.3f}")
        print(f"   Range: {summary['min_accuracy']:.3f} - {summary['max_accuracy']:.3f}")
        
        print(f"\n🔍 Component Scores:")
        print(f"   Alignment Score: {summary['avg_alignment_score']:.3f}")
        print(f"   Realism Score: {summary['avg_realism_score']:.3f}")
        print(f"   Preservation Score: {summary['avg_preservation_score']:.3f}")
        
        # Performance rating
        avg_accuracy = summary['avg_overall_accuracy']
        if avg_accuracy >= 0.8:
            rating = "🌟 Excellent"
        elif avg_accuracy >= 0.7:
            rating = "✅ Good"
        elif avg_accuracy >= 0.6:
            rating = "⚠️ Fair"
        else:
            rating = "❌ Needs Improvement"
        
        print(f"\n🏆 Overall Rating: {rating}")
        print("="*60)

def main():
    """Main function to run accuracy evaluation."""
    calculator = VirtualTryOnAccuracyCalculator()
    
    # Run evaluation
    results = calculator.run_batch_evaluation(sample_count=20)  # Start with 20 samples
    
    if results:
        # Save results
        filepath = calculator.save_results(results)
        
        # Generate report
        calculator.generate_report(results)
        
        print(f"\n📁 Detailed results saved to: {filepath}")
    else:
        print("❌ Evaluation failed")

if __name__ == "__main__":
    main()
