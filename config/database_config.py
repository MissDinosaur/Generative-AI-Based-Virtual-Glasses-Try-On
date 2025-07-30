"""
Database configuration for virtual try-on project.
"""
import os
from sqlalchemy import create_engine
import urllib.parse
import logging
import psycopg2
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv() # Load variables from .env
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_host = os.getenv("POSTGRES_HOST")
postgres_port = os.getenv("POSTGRES_PORT")
postgres_db = os.getenv("POSTGRES_DB")
postgres_schema = "diffusion"

class DatabaseConfig:
    """Database connection configuration."""
    
    def __init__(self):
        # Database credentials
        self.host = postgres_host
        self.port = postgres_port
        self.user = postgres_user
        self.password = postgres_password
        self.database = postgres_db
        self.schema = postgres_schema
        
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