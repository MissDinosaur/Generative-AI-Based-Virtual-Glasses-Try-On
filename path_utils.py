import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    return os.path.join(ROOT_DIR, 'data', filename)

def get_tryon_output_path():
    return os.path.join(ROOT_DIR, 'data', 'output')

def get_raw_selfies_path():
    return os.path.join(ROOT_DIR, 'data', 'raw', 'selfies')

def get_demo_output_path(img_name: str = ""):
    return os.path.join(ROOT_DIR, 'demo', 'output', img_name)

def get_demo_selfies_path(img_name: str = ""):
    return os.path.join(ROOT_DIR, 'demo', img_name)

def get_demo_glasses_path(img_name: str = ""):
    return os.path.join(ROOT_DIR, 'demo', "glasses", img_name)
