# Environment Setup

This page documents the complete setup process for the COS30018 stock prediction project environment.

## Prerequisites

- Python 3.12 or higher
- Git (for repository management)
- Sufficient disk space (~2GB for dependencies and data)

## Virtual Environment Setup

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd <path-to-project>

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (macOS/Linux)
source venv/bin/activate
```

### Step 2: Verify Installation

```bash
# Check Python version
python --version
# Expected output: Python 3.12

# Verify pip availability
pip --version
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify installation
pip list
```

## Required Dependencies

The project uses the following key packages:

```
numpy==2.3.2
pandas==2.3.2
matplotlib==3.10.5
tensorflow==2.20.0
scikit-learn==1.7.1
pandas-datareader==0.10.0
yfinance==0.2.65
joblib==1.5.1
```

## Configuration Details

- **Python Version**: 3.12
- **Tool Used**: Python's venv module
- **Location**: `<project-root>/venv`
- **Status**: Successfully activated and validated
- **Dependencies**: Installed from requirements.txt

## Verification Steps

The environment was verified through:

1. **Package Version Check**: `pip list` to confirm all packages installed
2. **Library Import Test**: Testing imports of key libraries (tensorflow, pandas, yfinance)
3. **TensorFlow Compatibility**: Confirming CPU optimization features
4. **Data Access Test**: Validating yfinance data retrieval functionality

## Hardware Optimization

The setup automatically detects and utilizes available CPU instructions:
- SSE3, SSE4.1, SSE4.2
- AVX, AVX2, AVX_VNNI
- FMA (Fused Multiply-Add)

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` for tensorflow
**Solution**: Ensure virtual environment is activated before installation

**Issue**: yfinance data download fails
**Solution**: Check internet connection and try different time periods

**Issue**: Memory errors during training
**Solution**: Reduce batch size in parameters.py

### Environment Validation

Run this test script to verify setup:

```python
# test_environment.py
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

print("✅ All imports successful")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")

# Test data download
try:
    data = yf.download("AAPL", period="5d")
    print("✅ yfinance data download successful")
except:
    print("❌ yfinance data download failed")
```

## Next Steps

After successful environment setup:

1. [Test v0.1 Model](v0.1-Model.md)
2. [Test P1 Model](P1-Model.md)
3. [Review Training Workflow](Training-Workflow.md)

---

*Environment successfully configured on: August 24, 2025*
