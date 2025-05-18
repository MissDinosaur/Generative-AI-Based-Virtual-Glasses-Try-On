"""Query data of glass frames from Postgres"""

from io import BytesIO
from sqlalchemy import create_engine
import pandas as pd
import requests
from PIL import Image
from dotenv import load_dotenv
import os
import path_utils

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

def download_frame_iamges(df:pd.DataFrame, title: str = "Ray-Ban 0RX6448 3086"):
    print("Start to download the images")
    row = df[df['title'] == title].iloc[0]
    main_img_url = row['main_image']
    raw_additional_imgs = row['additional_images']

    additional_img_urls = raw_additional_imgs.strip('{}').split(",")

    all_img_urls = [main_img_url] + additional_img_urls
    save_dir = path_utils.get_demo_glasses_path()

    for i, url in enumerate(all_img_urls):
        response = requests.get(url)
        response.raise_for_status()
        ext = url.split('.')[-1].split('?')[0] # extract the image extension, like jpg or png
        filename = f"image_{i+1}.{ext}"
        path = os.path.join(save_dir, filename)
        with open(path, 'wb') as f:
            f.write(response.content)
        #print(f"Downloaded: {path}")

if __name__ == "__main__":
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
        sample_df = pd.read_sql("SELECT title, main_image, additional_images FROM diffusion.frames LIMIT 10;", engine)
        #print("sample_df")
        # print(sample_df)

        download_frame_iamges(sample_df)
        print("Downloaded successfully.")

        # Save frame data into excel file
        # sample_df.to_excel('frame_data.xlsx', index=False)
        # print("CSV file saved successfully.")

        # print the value of additional_images
        # for i, val in enumerate(sample_df["additional_images"]):
        #     print(f"Row [{i}]: {val}")

        # Check the image given image url
        # main_img_urls = sample_df["main_image"]
        #check_image(main_img_urls[0])

    except Exception as e:
        print("Error occurred:", e)

