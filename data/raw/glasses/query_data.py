"""Query data of glass frames from Postgres"""

from io import BytesIO
from sqlalchemy import create_engine
import pandas as pd
import requests
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv() # Load variables from .env
user = os.getenv("POSTGRES_USER")
password = os.getenv("POSTGRES_PASSWORD")
host = os.getenv("POSTGRES_HOST")
port = os.getenv("POSTGRES_PORT")
db = os.getenv("POSTGRES_DB")

SCHEMA = "diffusion"
TABLE = ""

POSTGRES_URL = f"postgresql://{user}:{password}@{host}:{port}/{db}"

print(f"POSTGRES_URL: {POSTGRES_URL}")
engine = create_engine(POSTGRES_URL)

def check_image(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img.show()

def run_query(table: str):
    df = pd.read_sql(query, engine)
    print(df)
try:
    query_tables = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'diffusion'
    AND table_type = 'BASE TABLE';
    """
    
    query = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'diffusion'
    AND table_name = 'frames';
    """
    #df = pd.read_sql(query, engine)
    sample_df = pd.read_sql("SELECT title, main_image, additional_images FROM diffusion.frames LIMIT 5;", engine)
    main_img_urls = sample_df["main_image"]
    for i, val in enumerate(sample_df["additional_images"]):
        print(f"Row [{i}]: {val}")
    
    #check_image(main_img_urls[0])
    #print(sample_df)
except Exception as e:
    print("Error occurred:", e)

