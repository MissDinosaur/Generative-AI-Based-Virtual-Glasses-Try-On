"""
Download and extract SCUT-FBP5500 dataset from Google Drive.
"""
import gdown
import zipfile
import tempfile
import os
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Download and extract SCUT-FBP5500 dataset."""
    
    def __init__(self):
        self.file_id = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
        self.zip_filename = "SCUT-FBP5500_v2.1.zip"
        
    def download_and_extract(self, extract_to=None):
        """
        Download and extract dataset to temporary directory.
        
        Args:
            extract_to: Optional path to extract to. If None, uses temp directory.
            
        Returns:
            str: Path to extracted dataset directory
        """
        try:
            # Create temporary directory if not specified
            if extract_to is None:
                extract_to = tempfile.mkdtemp(prefix="scut_dataset_")
            else:
                os.makedirs(extract_to, exist_ok=True)
            
            logger.info(f"Downloading dataset to temporary location...")
            
            # Download file to temporary location
            temp_zip = os.path.join(extract_to, self.zip_filename)
            url = f"https://drive.google.com/uc?id={self.file_id}"
            
            gdown.download(url, temp_zip, quiet=False)
            logger.info(f"Downloaded {self.zip_filename}")
            
            # Extract zip file
            logger.info("Extracting dataset...")
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            
            # Remove zip file to save space
            os.remove(temp_zip)
            
            # Find the Images directory
            images_dir = None
            for root, dirs, files in os.walk(extract_to):
                if "Images" in dirs:
                    images_dir = os.path.join(root, "Images")
                    break
            
            if not images_dir:
                raise FileNotFoundError("Images directory not found in extracted dataset")
            
            logger.info(f"Dataset extracted to: {images_dir}")
            return images_dir
            
        except Exception as e:
            logger.error(f"Error downloading/extracting dataset: {e}")
            raise

# Global instance
dataset_downloader = DatasetDownloader()