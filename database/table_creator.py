"""
Create selfies table in PostgreSQL database.
"""
import logging
from config.database_config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableCreator:
    """Create and manage database tables."""
    
    def __init__(self):
        self.schema = db_config.schema
    
    def create_selfies_table(self):
        """Create selfies table if it doesn't exist."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            # Create table SQL
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.selfies (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                gender TEXT,
                race TEXT,
                image_data BYTEA NOT NULL,
                original_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            # Check if race column exists and add it if missing
            cursor.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{self.schema}' 
                AND table_name = 'selfies' 
                AND column_name = 'race';
            """)
            
            if not cursor.fetchone():
                logger.info("Adding missing 'race' column...")
                cursor.execute(f"ALTER TABLE {self.schema}.selfies ADD COLUMN race TEXT;")
                conn.commit()
                logger.info("Race column added successfully")
            
            logger.info(f"Table {self.schema}.selfies created successfully")
            
            # Create index for faster queries
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_selfies_filename 
            ON {self.schema}.selfies(filename);
            """
            cursor.execute(index_sql)
            conn.commit()
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating selfies table: {e}")
            raise
    
    def check_table_exists(self):
        """Check if selfies table exists."""
        try:
            conn = db_config.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = 'selfies'
                );
            """, (self.schema,))
            
            exists = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return exists
            
        except Exception as e:
            logger.error(f"Error checking table existence: {e}")
            return False

# Global instance
table_creator = TableCreator()
