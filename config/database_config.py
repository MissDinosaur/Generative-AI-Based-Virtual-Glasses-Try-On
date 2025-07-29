"""
Database configuration for virtual try-on project.
"""
import psycopg2
from sqlalchemy import create_engine
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database connection configuration."""
    
    def __init__(self):
        # Database credentials
        self.host = "152.53.12.68"
        self.port = 4000
        self.user = "student_diff"
        self.password = "diff_pass"
        self.database = "postgres"
        self.schema = "diffusion"
        
        # Create connection string
        encoded_password = urllib.parse.quote_plus(self.password)
        self.connection_string = f"postgresql://{self.user}:{encoded_password}@{self.host}:{self.port}/{self.database}"
        
        logger.info("Database configuration initialized")
    
    def get_connection(self):
        """Get psycopg2 connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
    
    def get_engine(self):
        """Get SQLAlchemy engine."""
        return create_engine(self.connection_string)

# Global instance
db_config = DatabaseConfig()