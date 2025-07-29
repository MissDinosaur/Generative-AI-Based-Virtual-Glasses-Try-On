import os
import zipfile
from collections import defaultdict
import gdown

# Step 1: Google Drive file info
GOOGLE_DRIVE_FILE_ID = "1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf"
OUTPUT_ZIP = "SCUT-FBP5500_v2.1.zip"
EXTRACT_DIR = "SCUT-FBP5500_v2.1"

# Step 2: Download if not already present
if not os.path.exists(OUTPUT_ZIP):
    url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
    print("📥 Downloading dataset from Google Drive...")
    gdown.download(url, OUTPUT_ZIP, quiet=False)
else:
    print("✅ Dataset already downloaded.")

# Step 3: Extract zip
if not os.path.exists(EXTRACT_DIR):
    print("🗂️ Extracting dataset...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
else:
    print("✅ Dataset already extracted.")

# Step 4: Locate the Images folder
images_root = None
for root, dirs, files in os.walk(EXTRACT_DIR):
    if "AF" in dirs and "AM" in dirs and "CF" in dirs and "CM" in dirs:
        images_root = root
        break

if not images_root:
    print("❌ Could not find 'Images' folder with all expected subfolders (AF, AM, CF, CM).")
else:
    # Mapping folders to gender/race
    category_map = {
        "AF": ("Asian", "Female"),
        "AM": ("Asian", "Male"),
        "CF": ("Caucasian", "Female"),
        "CM": ("Caucasian", "Male")
    }

    summary = defaultdict(list)

    for category in category_map:
        folder = os.path.join(images_root, category)
        if os.path.exists(folder):
            filenames = os.listdir(folder)
            image_files = [f for f in filenames if f.lower().endswith('.jpg')]
            summary[category].extend(image_files)
            print(f"✔️ {category}: {len(image_files)} images | {category_map[category][0]} {category_map[category][1]}")
            print(f"    Sample: {image_files[:3]}")
        else:
            print(f"❌ Folder not found: {folder}")

    total_images = sum(len(files) for files in summary.values())
    print(f"\n📦 Total images in dataset: {total_images}")
