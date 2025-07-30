# 🕶️ Virtual Glasses Try-On System

A sophisticated AI-powered virtual glasses try-on system that uses computer vision and machine learning to realistically overlay glasses on face images. The system integrates with PostgreSQL for data management and processes the SCUT-FBP5500 dataset for comprehensive testing.

## 🌟 Key Features

- **Real-time Virtual Try-On**: Advanced computer vision algorithms for realistic glasses overlay
- **Database Integration**: PostgreSQL storage for selfies and glasses data
- **Automated Dataset Processing**: Downloads and processes SCUT-FBP5500 dataset automatically
- **Accuracy Evaluation**: Comprehensive metrics to measure try-on quality
- **Batch Processing**: Handle multiple try-on combinations efficiently
- **Interactive Demo**: Easy-to-use demonstration interface

## 🎯 System Capabilities

### ✅ What the System Does
1. **Downloads SCUT-FBP5500 dataset** from Google Drive automatically
2. **Creates and manages database tables** for selfies and glasses
3. **Processes and stores** face images as binary data with metadata
4. **Fetches glasses** from existing frames database
5. **Performs virtual try-on** using MediaPipe face detection and OpenCV
6. **Evaluates accuracy** with multiple quality metrics
7. **Saves results** with organized file management

### 🔧 Technical Stack
- **Computer Vision**: OpenCV, MediaPipe
- **Database**: PostgreSQL with SQLAlchemy
- **Image Processing**: PIL, NumPy
- **Dataset**: SCUT-FBP5500 (5500 face images)
- **Languages**: Python 3.8+

## 🚀 Quick Start

### Prerequisites
```bash
# Ensure you have Python 3.8+ (<= 3.11) and PostgreSQL installed
python --version
psql --version
```

### 1. Setup Project
```bash
# Clone repository
git clone <your-repo-url>
cd virtual-tryon-project

# Create a virtual environment named <venvName> (Recommended Python version 3.11 or 3.10):
python -m venv <venvName>  # Replace <venvName> by your venv name 

# Activate the virtual environment <venvName>:
#Windows (CMD/Powershell):
<venvName>\Scripts\activate
# Windows (git bash): source <venvName>/Scripts/activate
# macOS/Linux: source <venvName>/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
python setup.py
```

### 2. Configure Database
Rename .env.example as .env and edit according to your PostgreSQL credentials:
```bash
POSTGRES_HOST = your_host_ip
POSTGRES_PORT = your_port
POSTGRES_USER = your_user_name
POSTGRES_PASSWORD = your_password
POSTGRES_DB = your_db_name
```

### 3. Run Demo
```bash
# Complete demo with dataset download
python demo/run_demo.py

# Quick data exploration
python avai_data.py

# Single try-on
python simple_pipeline.py --mode single

# Batch processing
python simple_pipeline.py --mode batch --batch-size 10

# Specific combinations
python simple_pipeline.py --mode single --selfie-id 1215 --glasses-id "689e52b6-3560-45c6-b2e0-a00182f4ab03"
```

## 📁 Project Architecture

```
virtual-tryon-project/
├── 📁 config/
│   └── database_config.py          # Database connection & configuration
│
├── 📁 data_processing/
│   ├── dataset_downloader.py       # SCUT dataset download & extraction
│   └── selfie_processor.py         # Image processing & database storage
│
├── 📁 database/
│   └── table_creator.py            # Database schema creation
│
├── 📁 core/
│   ├── virtual_tryon.py            # Main try-on algorithms
│   └── image_utils.py              # Image processing utilities
│
├── 📁 demo/
│   ├── run_demo.py                 # Complete program demonstration script
│   └── virtual_tryon_core.ipynb    # Core implementation of glasses try-on with runnable demo and adjustable functions
│
├── 📁 doc/                         # Detailed technical docs and project achitecture diagram
|
├── 📁 evaluation/
│   ├── accuracy_calculator.py      # Quality metrics & evaluation
│   └── results/                    # Evaluation reports
│
├── 📁 output/                     # Generated try-on results
│
├── setup.py                       # Project setup script
├── avai_data.py                   # Data exploration utility
├── simple_pipeline.py             # Command-line interface
├── .env.example
└── requirements.txt
```

## 🔬 Technical Deep Dive

### 1. Face Detection & Landmark Extraction
```python
# Uses MediaPipe for robust face detection
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)
```

**Key Points:**
- **468 facial landmarks** for precise feature detection
- **Eye region identification** for glasses positioning
- **Robust detection** across different face angles and lighting

### 2. Glasses Processing Pipeline
```python
def process_glasses_image(glasses_img):
    # 1. Background removal using alpha channel
    # 2. Edge cleaning and noise reduction
    # 3. Arm detection and removal
    # 4. Size normalization
    # 5. Quality validation
```

**Processing Steps:**
- **Alpha channel extraction** for transparency
- **Morphological operations** for edge cleaning
- **Contour analysis** for arm removal
- **Adaptive resizing** based on face dimensions

### 3. Advanced Overlay Algorithm
```python
def overlay_glasses(glasses, face, x, y):
    # 1. Region calculation and validation
    # 2. Alpha blending with edge smoothing
    # 3. Color correction for lighting consistency
    # 4. Gaussian blur for natural edges
    # 5. Final composition
```

**Overlay Features:**
- **Smart positioning** using eye landmarks
- **Alpha blending** for natural transparency
- **Color matching** to face lighting
- **Edge smoothing** for realistic appearance

### 4. Quality Evaluation Metrics

#### Alignment Accuracy (Weight: 40%)
- Measures how well glasses align with facial features
- Uses eye landmark consistency
- Calculates position and scale accuracy

#### Realism Score (Weight: 30%)
- **Sharpness**: Laplacian variance analysis
- **Contrast**: Standard deviation measurement
- **Brightness**: Consistency with original image
- **Edge preservation**: Canny edge detection
- **Noise level**: Bilateral filter comparison

#### Preservation Score (Weight: 30%)
- **Feature detection**: Glasses component identification
- **Color analysis**: Non-skin tone detection
- **Shape validation**: Aspect ratio verification

## 📊 Database Schema

### Selfies Table
```sql
CREATE TABLE diffusion.selfies (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    gender TEXT,                    -- Asian/Caucasian
    race TEXT,                      -- Male/Female
    image_data BYTEA NOT NULL,      -- Binary image data
    image_width INTEGER,            -- Image dimensions
    image_height INTEGER,
    original_path TEXT,             -- Source file path
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Frames Table (Existing)
```sql
-- Assumes existing table with:
-- id, brand, title, main_image, frame_shape, etc.
```

## 🎮 Usage Examples

### Basic Try-On
```python
from demo.run_demo import VirtualTryOnDemo

demo = VirtualTryOnDemo()
result = demo.run_single_tryon()
```

### Specific Combinations
```python
# Try specific selfie with random glasses
result = demo.run_single_tryon(selfie_id=123)

# Try specific glasses with random selfie
result = demo.run_single_tryon(glasses_id="uuid-string")

# Try specific combination
result = demo.run_single_tryon(selfie_id=123, glasses_id="uuid-string")
```

### Batch Processing
```python
# Process multiple random combinations
results = demo.run_batch_tryon(count=50)

# Command line batch processing
python simple_pipeline.py --mode batch --batch-size 20
```

### Accuracy Evaluation
```python
from evaluation.accuracy_calculator import VirtualTryOnAccuracyCalculator

calculator = VirtualTryOnAccuracyCalculator()
results = calculator.run_batch_evaluation(sample_count=100)
calculator.generate_report(results)
```

## 📈 Performance Metrics

### Latest Evaluation Results
```
🎯 VIRTUAL TRY-ON ACCURACY REPORT
============================================================
📊 Test Summary:
   Total Samples: 20
   Successful Runs: 20
   Success Rate: 100.0%

📈 Accuracy Metrics:
   Overall Accuracy: 0.793 ± 0.046
   Range: 0.730 - 0.900

🔍 Component Scores:
   Alignment Score: 0.986    # Excellent positioning
   Realism Score: 0.738      # Good image quality
   Preservation Score: 0.591  # Fair feature retention

🏆 Overall Rating: ✅ Good
```

## 🔍 Troubleshooting

### Common Issues

#### Database Connection
```bash
# Check PostgreSQL service
brew services start postgresql  # macOS
sudo service postgresql start   # Linux

# Test connection
python -c "from config.database_config import db_config; print(db_config.test_connection())"
```

#### Missing Dependencies
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt

# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"
```

#### Dataset Download Issues
```bash
# Manual download if automatic fails
python -c "from data_processing.dataset_downloader import dataset_downloader; dataset_downloader.download_and_extract()"
```

#### Memory Issues
```bash
# Reduce batch size for large datasets
python simple_pipeline.py --mode batch --batch-size 5
```

### Error Codes
- **Error 001**: Database connection failed → Check credentials
- **Error 002**: No glasses found → Ensure frames table has data
- **Error 003**: Face detection failed → Check image quality
- **Error 004**: Overlay failed → Check glasses image format

## 🛠️ Development

### Adding New Features
1. **New Glasses Sources**: Extend `data_processing/` modules
2. **Custom Metrics**: Add to `evaluation/accuracy_calculator.py`
3. **UI Improvements**: Modify `demo/` scripts
4. **Database Changes**: Update `database/table_creator.py`

### Testing
```bash
# Run accuracy evaluation
python evaluation/accuracy_calculator.py

# Test specific components
python -m pytest tests/  # If tests are added

# Manual testing
python avai_data.py
python simple_pipeline.py --mode single
```

### Performance Optimization
- **Batch Processing**: Use `run_batch_tryon()` for multiple images
- **Database Indexing**: Add indexes on frequently queried columns
- **Image Caching**: Cache processed glasses images
- **Parallel Processing**: Use multiprocessing for large batches

## 📚 Technical References

### Computer Vision Concepts
- **MediaPipe Face Mesh**: 468-point facial landmark detection
- **Alpha Blending**: Transparent overlay techniques
- **Morphological Operations**: Image cleaning and processing
- **Contour Analysis**: Shape detection and filtering

### Database Design
- **Binary Storage**: Efficient image storage in PostgreSQL
- **Schema Design**: Normalized tables for scalability
- **Indexing Strategy**: Optimized queries for large datasets

### Quality Metrics
- **Laplacian Variance**: Sharpness measurement
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Canny Edge Detection**: Edge preservation analysis
- **Bilateral Filtering**: Noise estimation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 🙏 Acknowledgments

- **SCUT-FBP5500 Dataset**: Face beauty prediction dataset
- **MediaPipe**: Google's ML framework for face detection
- **OpenCV**: Computer vision library
- **PostgreSQL**: Robust database system

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed description
4. Include error logs and system information

---

## 📥 Dataset Download

The SCUT-FBP5500 dataset (171MB) is automatically downloaded when you run the demo:

```bash
# Dataset will be downloaded automatically
python demo/run_demo.py
```

**Note**: The dataset is not included in the repository due to size constraints. It will be downloaded from Google Drive on first run.

