"""
Setup script for virtual try-on project.
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ Requirements installed")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = [
        "output",
        "logs",
        "temp"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")

def main():
    """Main setup function."""
    print("🚀 Setting up Virtual Try-On Project...")
    
    install_requirements()
    create_directories()
    
    print("\n✅ Setup complete!")
    print("\nTo run the demo:")
    print("python demo/run_demo.py")

if __name__ == "__main__":
    main()
