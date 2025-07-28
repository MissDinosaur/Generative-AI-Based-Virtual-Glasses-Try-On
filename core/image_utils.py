"""
Image utility functions for virtual try-on.
"""
import cv2
import numpy as np
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def save_binary_to_temp(image_data):
        """
        Save binary image data to temporary file.
        
        Args:
            image_data: Binary image data from database
            
        Returns:
            str: Path to temporary image file
        """
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            
            # Write binary data to file
            with os.fdopen(temp_fd, 'wb') as f:
                f.write(image_data)
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Error saving binary to temp file: {e}")
            raise
    
    @staticmethod
    def cleanup_temp_file(temp_path):
        """Remove temporary file."""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {temp_path}: {e}")
    
    @staticmethod
    def validate_image(image_path):
        """Validate that image can be loaded."""
        try:
            img = cv2.imread(image_path)
            return img is not None
        except:
            return False

# Global instance
image_utils = ImageUtils()