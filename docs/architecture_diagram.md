# 🏗️ Virtual Glasses Try-On System Architecture

## 📊 System Overview
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        VIRTUAL GLASSES TRY-ON SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DATA LAYER    │    │  PROCESSING     │    │   CORE ENGINE   │    │   OUTPUT LAYER  │
│                 │    │     LAYER       │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ PostgreSQL  │ │    │ │ MediaPipe   │ │    │ │ Virtual     │ │    │ │ Generated   │ │
│ │ Database    │ │◄───┤ │ Face        │ │◄───┤ │ Try-On      │ │───►│ │ Images      │ │
│ │             │ │    │ │ Detection   │ │    │ │ Engine      │ │    │ │             │ │
│ │ • Selfies   │ │    │ │             │ │    │ │             │ │    │ │ • Results   │ │
│ │ • Frames    │ │    │ │ OpenCV      │ │    │ │ • Overlay   │ │    │ │ • Metrics   │ │
│ └─────────────┘ │    │ │ Processing  │ │    │ │ • Blending  │ │    │ │ • Reports   │ │
└─────────────────┘    │ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
                       └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 Data Flow Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────────┘

1. DATA INGESTION
   ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   │ SCUT Dataset    │    │ Dataset         │    │ Selfie          │
   │ (Google Drive)  │───►│ Downloader      │───►│ Processor       │
   │ 171MB ZIP       │    │ • Download      │    │ • Extract       │
   └─────────────────┘    │ • Extract       │    │ • Categorize    │
                          │ • Validate      │    │ • Store Binary  │
                          └─────────────────┘    └─────────────────┘
                                                          │
                                                          ▼
2. DATABASE STORAGE                                ┌─────────────────┐
   ┌─────────────────────────────────────────────► │ PostgreSQL DB   │
   │                                               │ • selfies table │
   │  ┌─────────────────┐    ┌─────────────────┐   │ • frames table  │
   │  │ Frames Data     │    │ Table Creator   │   │ • Binary data   │
   │  │ (Existing)      │───►│ • Schema setup  │ ──┤ • Metadata      │
   │  │ • Brand/Title   │    │ • Indexes       │   └─────────────────┘
   │  │ • Images        │    │ • Constraints   │
   │  └─────────────────┘    └─────────────────┘
   │
3. PROCESSING PIPELINE
   │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
   └─►│ Face Detection  │───►│ Landmark        │───►│ Glasses         │
      │ • MediaPipe     │    │ Extraction      │    │ Processing      │
      │ • 468 points    │    │ • Eye regions   │    │ • Alpha channel │
      │ • Confidence    │    │ • Positioning   │    │ • Arm removal   │
      └─────────────────┘    └─────────────────┘    └─────────────────┘
                                                              │
                                                              ▼
4. VIRTUAL TRY-ON ENGINE                            ┌─────────────────┐
   ┌─────────────────┐    ┌─────────────────┐       │ Overlay Engine  │
   │ Image Alignment │◄───│ Core Algorithm  │◄──────│ • Alpha blend   │
   │ • Scale/Rotate  │    │ • Position calc │       │ • Edge smooth   │
   │ • Perspective   │    │ • Size adjust   │       │ • Color match   │
   └─────────────────┘    └─────────────────┘       └─────────────────┘
                                   │
                                   ▼
5. OUTPUT & EVALUATION            ┌─────────────────┐
   ┌─────────────────┐            │ Result Image    │
   │ Quality Metrics │◄───────────│ • Composite     │
   │ • Alignment     │            │ • Realistic     │
   │ • Realism       │            │ • High quality  │
   │ • Preservation  │            └─────────────────┘
   └─────────────────┘
```

## 🏛️ Component Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            COMPONENT BREAKDOWN                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

CONFIG LAYER
├── database_config.py ──────► PostgreSQL connection & credentials

DATA PROCESSING LAYER
├── dataset_downloader.py ───► SCUT dataset management
├── selfie_processor.py ─────► Image processing & DB storage
└── table_creator.py ────────► Database schema management

CORE ENGINE
├── virtual_tryon.py ────────► Main try-on algorithms
│   ├── Face detection (MediaPipe)
│   ├── Landmark extraction (468 points)
│   ├── Glasses processing (Alpha, ARM removal)
│   ├── Overlay algorithm (Blending, positioning)
│   └── Result generation
└── image_utils.py ──────────► Utility functions

DEMO & INTERFACE
├── run_demo.py ─────────────► Complete demonstration
└── simple_pipeline.py ──────► CLI commands

EVALUATION SYSTEM
└── accuracy_calculator.py ──► Quality metrics & reporting
│   ├── Alignment accuracy (~90%)
│   ├── Realism score (~70%)
│   └── Preservation score (~50%)
└── Overall accuracy (~75%) ──────────► Utility functions
```

## 🔧 Technology Stack
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TECHNOLOGY STACK                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

COMPUTER VISION
├── MediaPipe ───────────────► Face detection & landmarks
├── OpenCV ──────────────────► Image processing & manipulation
└── PIL/NumPy ───────────────► Image handling & arrays

DATABASE
├── PostgreSQL ──────────────► Primary data storage
├── SQLAlchemy ──────────────► ORM & database abstraction
└── Binary Storage ──────────► Efficient image storage

MACHINE LEARNING
├── Face Mesh (468 points) ──► Precise facial feature detection
├── Alpha Blending ──────────► Realistic overlay techniques
└── Quality Metrics ─────────► Automated evaluation

PYTHON ECOSYSTEM
├── Python 3.8+ ─────────────► Core language
├── Virtual Environment ─────► Dependency isolation
└── Requirements.txt ────────► Package management
```