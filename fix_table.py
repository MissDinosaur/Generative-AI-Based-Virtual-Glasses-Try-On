"""
Quick fix script to add missing race column to existing selfies table.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.database_config import db_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_selfies_table():
    """Add missing race column to selfies table."""
    try:
        conn = db_config.get_connection()
        cursor = conn.cursor()
        
        # Check if race column exists
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{db_config.schema}' 
            AND table_name = 'selfies' 
            AND column_name = 'race';
        """)
        
        if not cursor.fetchone():
            logger.info("Adding missing 'race' column...")
            cursor.execute(f"ALTER TABLE {db_config.schema}.selfies ADD COLUMN race TEXT;")
            conn.commit()
            logger.info("✅ Race column added successfully")
        else:
            logger.info("Race column already exists")
        
        # Check if gender column exists
        cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = '{db_config.schema}' 
            AND table_name = 'selfies' 
            AND column_name = 'gender';
        """)
        
        if not cursor.fetchone():
            logger.info("Adding missing 'gender' column...")
            cursor.execute(f"ALTER TABLE {db_config.schema}.selfies ADD COLUMN gender TEXT;")
            conn.commit()
            logger.info("✅ Gender column added successfully")
        else:
            logger.info("Gender column already exists")
        
        cursor.close()
        conn.close()
        
        logger.info("✅ Table structure updated successfully!")
        
    except Exception as e:
        logger.error(f"Error fixing table: {e}")
        raise

if __name__ == "__main__":
    fix_selfies_table()