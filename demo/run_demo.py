"""
Main demo script for virtual try-on system.
This script orchestrates the entire process.
"""
import logging
import random
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.database_config import db_config
from data_processing.dataset_downloader import dataset_downloader
from data_processing.selfie_processor import selfie_processor
from database.table_creator import table_creator
from core.virtual_tryon import main_tryon_from_binary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualTryOnDemo:
    """Main demo class for virtual try-on system."""
    
    def __init__(self):
        self.schema = db_config.schema
        
    def setup_database(self):
        """Setup database tables."""
        logger.info("🔧 Setting up database...")
        
        # Create selfies table
        table_creator.create_selfies_table()
        logger.info("✅ Database setup complete")
    
    def get_table_columns(self):
        """Get available columns in selfies table."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{self.schema}' 
                AND table_name = 'selfies'
                ORDER BY ordinal_position;
            """)
            
            columns = [row[0] for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            return columns
            
        except Exception as e:
            logger.error(f"Error getting table columns: {e}")
            return ['id', 'filename', 'image_data']  # Minimum required columns
    
    def download_and_process_dataset(self, limit=100):
        """Download dataset and process selfies."""
        logger.info("📥 Downloading and processing SCUT dataset...")
        
        try:
            # Download and extract dataset
            images_dir = dataset_downloader.download_and_extract()
            
            # Process and store selfies
            selfie_processor.process_and_store_selfies(images_dir, limit=limit)
            
            logger.info("✅ Dataset processing complete")
            
        except Exception as e:
            logger.error(f"Error in dataset processing: {e}")
            raise
    
    def get_random_selfie(self):
        """Get a random selfie from database."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Get available columns
            columns = self.get_table_columns()
            
            # Build SELECT query based on available columns
            select_columns = ['id', 'filename']
            if 'gender' in columns:
                select_columns.append('gender')
            if 'race' in columns:
                select_columns.append('race')
            select_columns.append('image_data')
            
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM {self.schema}.selfies 
                ORDER BY RANDOM() 
                LIMIT 1
            """
            
            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                selfie_data = {
                    'id': result[0],
                    'filename': result[1],
                    'gender': result[2] if len(result) > 3 and 'gender' in columns else 'Unknown',
                    'race': result[3] if len(result) > 4 and 'race' in columns else 'Unknown',
                    'image_data': result[-1]  # Always last column
                }
                return selfie_data
            else:
                raise ValueError("No selfies found in database")
                
        except Exception as e:
            logger.error(f"Error getting random selfie: {e}")
            raise
    
    def get_random_glasses(self):
        """Get random glasses from frames table."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT id, brand, title, main_image 
                FROM {self.schema}.frames 
                WHERE main_image IS NOT NULL 
                ORDER BY RANDOM() 
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'brand': result[1] if result[1] else 'Unknown Brand',
                    'title': result[2] if result[2] else 'Unknown Model',
                    'main_image': result[3]
                }
            else:
                raise ValueError("No glasses found in database")
                
        except Exception as e:
            logger.error(f"Error getting random glasses: {e}")
            raise
    
    def get_specific_selfie(self, selfie_id):
        """Get specific selfie by ID."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Get available columns
            columns = self.get_table_columns()
            
            # Build SELECT query based on available columns
            select_columns = ['id', 'filename']
            if 'gender' in columns:
                select_columns.append('gender')
            if 'race' in columns:
                select_columns.append('race')
            select_columns.append('image_data')
            
            query = f"""
                SELECT {', '.join(select_columns)}
                FROM {self.schema}.selfies 
                WHERE id = %s
            """
            
            cursor.execute(query, (selfie_id,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                selfie_data = {
                    'id': result[0],
                    'filename': result[1],
                    'gender': result[2] if len(result) > 3 and 'gender' in columns else 'Unknown',
                    'race': result[3] if len(result) > 4 and 'race' in columns else 'Unknown',
                    'image_data': result[-1]  # Always last column
                }
                return selfie_data
            else:
                raise ValueError(f"Selfie with ID {selfie_id} not found")
                
        except Exception as e:
            logger.error(f"Error getting specific selfie: {e}")
            raise
    
    def get_specific_glasses(self, glasses_id):
        """Get specific glasses by ID."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT id, brand, title, main_image 
                FROM {self.schema}.frames 
                WHERE id = %s
            """, (glasses_id,))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'brand': result[1] if result[1] else 'Unknown Brand',
                    'title': result[2] if result[2] else 'Unknown Model',
                    'main_image': result[3]
                }
            else:
                raise ValueError(f"Glasses with ID {glasses_id} not found")
                
        except Exception as e:
            logger.error(f"Error getting specific glasses: {e}")
            raise
    
    def run_single_tryon(self, selfie_id=None, glasses_id=None, save_result=True):
        """Run a single virtual try-on."""
        try:
            logger.info("🚀 Starting virtual try-on demo...")
            
            # Get selfie
            if selfie_id:
                selfie = self.get_specific_selfie(selfie_id)
            else:
                selfie = self.get_random_selfie()
            
            # Get glasses
            if glasses_id:
                glasses = self.get_specific_glasses(glasses_id)
            else:
                glasses = self.get_random_glasses()
            
            logger.info(f"👤 Selfie: {selfie['filename']} ({selfie['gender']}, {selfie['race']})")
            logger.info(f"👓 Glasses: {glasses['brand']} - {glasses['title']}")
            
            # Prepare save path
            save_path = None
            if save_result:
                output_dir = project_root / "output"
                output_dir.mkdir(exist_ok=True)
                save_path = str(output_dir / f"tryon_{selfie['id']}_{glasses['id']}.jpg")
            
            # Run virtual try-on
            result_img = main_tryon_from_binary(
                selfie['image_data'],
                glasses['main_image'],
                save_result=save_path
            )
            
            logger.info("✅ Virtual try-on completed successfully!")
            
            return {
                'success': True,
                'selfie': selfie,
                'glasses': glasses,
                'result_path': save_path
            }
            
        except Exception as e:
            logger.error(f"Error in virtual try-on: {e}")
            return {'success': False, 'error': str(e)}
    
    def run_batch_tryon(self, count=5):
        """Run multiple virtual try-ons."""
        logger.info(f"🔄 Running {count} virtual try-ons...")
        
        results = []
        for i in range(count):
            logger.info(f"Processing {i+1}/{count}...")
            result = self.run_single_tryon()
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"✅ Batch complete: {successful}/{count} successful")
        
        return results
    
    def check_data_status(self):
        """Check status of data in database."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Check selfies count
            cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.selfies")
            selfies_count = cursor.fetchone()[0]
            
            # Check frames count
            cursor.execute(f"SELECT COUNT(*) FROM {self.schema}.frames WHERE main_image IS NOT NULL")
            frames_count = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
            logger.info(f"📊 Data Status:")
            logger.info(f"   Selfies: {selfies_count}")
            logger.info(f"   Glasses: {frames_count}")
            
            return {'selfies': selfies_count, 'glasses': frames_count}
            
        except Exception as e:
            logger.error(f"Error checking data status: {e}")
            return {'selfies': 0, 'glasses': 0}

def main():
    """Main function to run the complete demo."""
    demo = VirtualTryOnDemo()
    
    try:
        # Step 1: Setup database
        demo.setup_database()
        
        # Step 2: Check current data status
        status = demo.check_data_status()
        
        # Step 3: Download and process dataset if needed
        if status['selfies'] == 0:
            logger.info("No selfies found. Downloading dataset...")
            demo.download_and_process_dataset(limit=50)  # Limit for demo
        else:
            logger.info(f"Found {status['selfies']} selfies in database")
        
        # Step 4: Check if we have glasses data
        if status['glasses'] == 0:
            logger.warning("No glasses found in database. Please ensure frames table has data.")
            return
        
        # Step 5: Run demo
        print("\n" + "="*50)
        print("🎉 VIRTUAL TRY-ON DEMO")
        print("="*50)
        
        # Single try-on
        demo.run_single_tryon()
        
        # Ask user if they want more
        while True:
            choice = input("\nWould you like to try another combination? (y/n): ").lower()
            if choice == 'y':
                demo.run_single_tryon()
            elif choice == 'n':
                break
            else:
                print("Please enter 'y' or 'n'")
        
        logger.info("🎉 Demo completed!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise

if __name__ == "__main__":
    main()
