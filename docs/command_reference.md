# 🎮 Command Reference Guide

## Basic Commands

### Single Try-On
```bash
# Random combination
python simple_pipeline.py --mode single

# Specific selfie (random glasses)
python simple_pipeline.py --mode single --selfie-id 1081

# Specific glasses (random selfie)
python simple_pipeline.py --mode single --glasses-id "ca8a573e-3549-46b7-8d9d-8d84f336199f"

# Specific combination
python simple_pipeline.py --mode single --selfie-id 1081 --glasses-id "ca8a573e-3549-46b7-8d9d-8d84f336199f"
```

### Batch Processing
```bash
# Generate 5 random combinations
python simple_pipeline.py --mode batch --batch-size 5

# Generate 20 combinations
python simple_pipeline.py --mode batch --batch-size 20

# Large batch (50 combinations)
python simple_pipeline.py --mode batch --batch-size 50
```

### Data Exploration
```bash
# See available data
python avai_data.py

# Interactive demo
python demo/simple_demo.py

# Complete demo with dataset download
python demo/run_demo.py
```

### Evaluation & Testing
```bash
# Run accuracy evaluation (20 samples)
python evaluation/accuracy_calculator.py

# Custom evaluation count
python -c "
from evaluation.accuracy_calculator import VirtualTryOnAccuracyCalculator
calc = VirtualTryOnAccuracyCalculator()
results = calc.run_batch_evaluation(sample_count=50)
calc.generate_report(results)
"
```

## Advanced Usage

### Custom Parameters
```bash
# Adjust glasses scaling
python -c "
from demo.run_demo import VirtualTryOnDemo
demo = VirtualTryOnDemo()
demo.run_single_tryon()  # Uses default parameters
"
```

### Programmatic Access
```python
# In Python script
from demo.run_demo import VirtualTryOnDemo

demo = VirtualTryOnDemo()

# Single try-on with specific IDs
result = demo.run_single_tryon(selfie_id=123, glasses_id="uuid")

# Batch processing
results = demo.run_batch_tryon(count=10)
```