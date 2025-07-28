"""
Process and store selfies from SCUT dataset into PostgreSQL.
"""
import os
import cv2
import logging
from pathlib import Path
from tqdm import tqdm
from config.database_config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfieProcessor:
    """Process and store selfies in database."""
    
    def __init__(self):
        self.schema = db_config.schema
        
    def extract_metadata(self, filename, folder_name):
        """Extract gender and race from filename and folder."""
        # Extract from folder name: AF, AM, CF, CM
        if folder_name.startswith('AF'):
            gender, race = 'Female', 'Asian'
        elif folder_name.startswith('AM'):
            gender, race = 'Male', 'Asian'
        elif folder_name.startswith('CF'):
            gender, race = 'Female', 'Caucasian'
        elif folder_name.startswith('CM'):
            gender, race = 'Male', 'Caucasian'
        else:
            gender, race = 'Unknown', 'Unknown'
            
        return gender, race
    
    def process_and_store_selfies(self, images_dir, limit=None):
        """
        Process selfies from dataset and store in database.
        
        Args:
            images_dir: Path to Images directory
            limit: Optional limit on number of images to process
        """
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Get all image files
            image_files = []
            folders = ['AF', 'AM', 'CF', 'CM']
            
            for folder in folders:
                folder_path = os.path.join(images_dir, folder)
                if os.path.exists(folder_path):
                    for filename in os.listdir(folder_path):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_files.append((folder, filename, os.path.join(folder_path, filename)))
            
            if limit:
                image_files = image_files[:limit]
            
            logger.info(f"Processing {len(image_files)} selfie images...")
            
            processed_count = 0
            skipped_count = 0
            
            for folder, filename, filepath in tqdm(image_files, desc="Processing selfies"):
                try:
                    # Check if already exists
                    cursor.execute(f"SELECT id FROM {self.schema}.selfies WHERE filename = %s", (filename,))
                    if cursor.fetchone():
                        skipped_count += 1
                        continue
                    
                    # Read image as binary
                    with open(filepath, 'rb') as f:
                        image_data = f.read()
                    
                    # Extract metadata
                    gender, race = self.extract_metadata(filename, folder)
                    original_path = f"Images/{folder}/{filename}"
                    
                    # Insert into database
                    insert_sql = f"""
                    INSERT INTO {self.schema}.selfies (filename, gender, race, image_data, original_path)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    
                    cursor.execute(insert_sql, (filename, gender, race, image_data, original_path))
                    processed_count += 1
                    
                    # Commit every 50 records
                    if processed_count % 50 == 0:
                        conn.commit()
                        
                except Exception as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue
            
            # Final commit
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Processing complete! Processed: {processed_count}, Skipped: {skipped_count}")
            
        except Exception as e:
            logger.error(f"Error processing selfies: {e}")
            raise

# Global instance
selfie_processor = SelfieProcessor()