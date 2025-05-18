## Generative AI-Based Virtual Glasses Try-On (Static-photo-based)
### Project Introduction
This is a realistic image-based system (generative model).

**Input**

One pair of glasses: It would have multiple images of different angles.

One selfie: The head in the image may has an angle 

**Goal/output**: Selecting the glasses image with the most appropriate angel and overlay it on the selfie with a realistic effect. In another word, generate realistic images of a person wearing glasses — not just simple overlaying. 

#### Key Concepts:
1. Image Fusion Instead of Overlay:

    The goal is to make the glasses blend naturally with the face — respecting occlusion (e.g., glasses behind ears), lighting consistency, and facial perspective.

    Tech: StyleGAN, Diffusion Models (e.g., ControlNet, Stable Diffusion), or Image-to-Image models.

2. Face Landmark Detection & 3D Alignment:

    Use face landmark detection (e.g., MediaPipe Face Mesh or dlib) to determine where the glasses should be placed.

    Optionally, extract 3D facial geometry using tools like 3D Morphable Models or DECA to match the glasses’ angle with the face orientation.

3. Image Generation Pipeline:

    Input: face image + glasses image.

    Model: Use a UNet with attention or a diffusion-based model trained to output naturally merged images.

    Use inpainting models to handle occlusion, fine edges (e.g., glass frame over the nose), and lighting consistency.

4. Training Data:

    Either use the dataset of selfies + glasses pairs (if labeled).

    Or pretrain on public datasets (e.g., CelebA + synthetic glasses) and fine-tune on your data using transfer learning.

### Project Achitecture
```text
genai-virtual-glasses-tryon/
│
├── app/                     # Core backend logic
│   ├── __init__.py
│   ├── config.py              # Paths, constants, model settings
│   ├── routes.py              # Main backend logic (Flask request handling & AR interaction)
│   ├── utils/
│   └── ui                     # Flask frontend 
│       ├── templates/             # HTML upload page or try-on viewer
│       └── static/                # CSS, JS, webcam script, glasses
|
│── Image_processing
│   ├── download_selfies.py    # Download selfies from the open Google Drive sharing link
│   ├── image_io.py            # Load/save images
│   ├── face_utils.py          # Face detection, MediaPipe landmarks extraction
│   ├── overlay.py             # Glasses-face alignment logic, overplaying glasses on the selfie
|   └── gen_ai/              # Image Fusion by Gen-AI
│       ├── model_wrapper.py   # Train/Load the generative model, eg. Diffusers (HuggingFace) or ControlNet. 
│       └── controlnet.py      # ControlNet or other custom model
|
├── data/                    # Data assets and structure
│   ├── raw/                   # Original selfies & glasses
│   ├── processed/             # Aligned faces, resized glasses
│   ├── landmarks/             # Facial keypoint files
│   └── output/                # Try-on result images (face + glasses)
│
├── notebooks/               # Prototyping notebooks, for early-stage testing
│   ├── landmarks_explore.ipynb
│   └── model_experiment.ipynb
|
├── demo/                    # Simple demo for images overlay
│   ├── demo_tryon.py          # Scripts to run the demo
│
├── main.py                  # Script to start app or model
├── path_utils.py            # Path management
├── README.md
└── requirements.txt         # Python dependency list

```

### Tech Stack
Frontend : 
Backend: Flask (serves config, assets, or logs)
Facial Tracking: Use MediaPipe Face Mesh for 3D facial landmarks
Overlay Glasses: Diffusers (HuggingFace) or ControlNet

### Datasource
1. Selfie data is from https://github.com/HCIILAB/SCUT-FBP5500-Database-Release?tab=readme-ov-file

2. Glasses frams data is provides by Pickz-AI and is accessed through Postgres


###  Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-folder-name>
   ```

2. Create a virtual environment named <venvName> (Recommended Python version 3.11 or 3.10):
   ```bash
   python -m venv <venvName>  # Replace <venvName> by your venv name 
   ```

3. Activate the virtual environment <venvName>:
   ```bash
   # Replace <venvName> by your venv name 

   # Windows (CMD/Powershell)
   <venvName>\Scripts\activate

   # Windows (git bash)
   source <venvName>/Scripts/activate

   # macOS/Linux
   source <venvName>/bin/activate

   # if you wanna quit the current virtual environment
   deactivate
   ```

4. Install the dependencies (Recommended Python version 3.11 or 3.10):
   ```bash
   pip install -r requirements.txt
   ```