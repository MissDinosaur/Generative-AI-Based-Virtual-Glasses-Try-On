import os
import gdown
import zipfile
from dotenv import load_dotenv
import path_utils
from glob import glob

load_dotenv() # Load variables from .env

# set up the downloading link
file_id = os.getenv("GOOGLE_DRIVE_FILE_ID")
zip_url = f"https://drive.google.com/uc?id={file_id}"
zip_file_name = os.getenv("GOOGLE_DRIVE_ZIP_FILE_NAME")
print(f"zip_url: {zip_url}")
print(f"zip_file_name: {zip_file_name}")

# set up the local path for the zip file
extract_folder = path_utils.get_raw_selfies_path() # "data/raw/selfies"
print(f"extract_folder: {extract_folder}")

# download the zip file from Google Drive
if not os.path.exists(zip_file_name):
    print("Downloading zip file from Google Drive...")
    gdown.download(zip_url, zip_file_name, quiet=False)
else:
    print("Zip file already exists, skipping download.")

# Only extract the images under SCUT-FBP5500_v2/Images folder
SELECTED_FOLDER = "SCUT-FBP5500_v2/Images"
with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
    for file in zip_ref.namelist():
        if f"{SELECTED_FOLDER}/" in file and file.lower().endswith((".jpg", ".jpeg", ".png")):
            zip_ref.extract(file, extract_folder)

print("Done. Images extracted to:", os.path.join(extract_folder, SELECTED_FOLDER))

image_files = glob(os.path.join(extract_folder, SELECTED_FOLDER, "*.jpg"))
 
# AF1-AR2000， AM1-2000， CF1-CF750， CM1-CM750
print(f"And total {len(image_files)} images are extracted") # 5500 in total