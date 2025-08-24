```markdown
# COS30018 - Option C - Task 1: Setup and Model Evaluation Report

**Student:** Chiraath Madahapola
**Date:** August 24, 2025  
**Course:** COS30018 - Intelligent Systems  

## Executive Overview

This document evaluates two distinct stock prediction systems, the initial v0.1 model and the upgraded P1 model, both employing LSTM neural networks. The P1 version excels with advanced data management, model storage capabilities, and a broader range of performance indicators.

## 1. Environment Configuration

### 1.1 Virtual Environment Setup and Settings

#### 1.1.1 Process for Setting Up the Virtual Environment
The project environment was established using Python's `venv` to maintain isolated dependencies:

**Step 1: Creating the Virtual Environment**
```bash
# Move to the project folder
cd C:\Users\user\Desktop\int sys\option_c\project_option_c

# Initialize a new virtual environment
python -m venv venv

# Start the virtual environment (Windows)
venv\Scripts\activate
```

**Step 2: Checking the Environment**
```bash
# Check Python version
python --version
# Expected: Python 3.12.3

# Check pip version
pip --version
```

**Step 3: Installing Dependencies**
```bash
# Install packages listed in requirements.txt
pip install -r requirements.txt

# Confirm installation
pip list
```

#### 1.1.2 Configuration Specifications
- **Python Version:** 3.12.3
- **Tool Used:** venv
- **Location:** C:\Users\user\Desktop\int sys\option_c\project_option_c\venv
- **Activation Status:** Confirmed active
- **Dependencies:** Loaded from `requirements.txt`

### 1.2 Necessary Dependencies
Installed packages include:
```
numpy==1.26.4
matplotlib==3.6.3
pandas==2.3.0
tensorflow==2.19.0
scikit-learn==1.6.1
yfinance==0.2.65
joblib==1.5.1
```

### 1.3 Verification of Setup
Validation included:
- Package version review with `pip list`
- Module import tests
- TensorFlow CPU capability check
- yfinance data retrieval test

The setup is fully operational for both models.

## 2. Model Descriptions

### 2.1 v0.1 Model Description
A foundational LSTM-based predictor:

**Key Attributes:**
- **Stock:** CBA.AX
- **Training Period:** 2020-01-01 to 2023-08-01
- **Testing Period:** 2023-08-02 to 2024-07-02
- **Structure:** 3-layer LSTM with dropout
- **Forecast Window:** 60-day lookback, 1-day forecast
- **Data Source:** yfinance
- **Features:** Close price only

**Model Structure:**
- LSTM Layer 1: 50 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 50 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 3: 50 units
- Dropout: 0.2
- Dense Output: 1 unit

### 2.2 P1 Model Description
An enhanced LSTM model with added functionalities:

**Key Attributes:**
- **Stock:** META
- **Training Period:** 5 years
- **Structure:** 2-layer LSTM with adjustable settings
- **Forecast Window:** 50-day lookback, 15-day forecast
- **Data Source:** yfinance
- **Features:** adjclose, volume, open, high, low

**Model Structure:**
- LSTM Layer 1: 256 units, return_sequences=True
- Dropout: 0.4
- LSTM Layer 2: 256 units
- Dropout: 0.4
- Dense Output: 1 unit

## 3. Model Training and Evaluation Process

### 3.1 v0.1 Model Training and Evaluation

#### 3.1.1 Training Procedure
Run with:
```bash
python C:\Users\user\Desktop\int sys\option_c\project_option_c\v0.1.py
```

**Training Settings:**
- **Data Retrieval:** yfinance for CBA.AX
- **Training Period:** 2020-01-01 to 2023-08-01
- **Testing Period:** 2023-08-02 to 2024-07-02
- **Preprocessing:** MinMaxScaler (0,1 range)
- **Sequence Length:** 60 days
- **Batch Size:** 32
- **Epochs:** 25
- **Optimizer:** Adam
- **Loss Function:** MSE

**Training Output:**
```
Epoch 1/25: loss: 0.0797
Epoch 2/25: loss: 0.0114
...
Epoch 25/25: loss: 0.0048
```

#### 3.1.2 Evaluation Procedure
- **Data Prep:** Merged train and test datasets
- **Prediction:** Generated test predictions
- **Scaling Back:** Reverted to original price scale
- **Visualization:** Plotted actual vs predicted
- **Next Day Forecast:** Single prediction

### 3.2 P1 Model Training and Evaluation

#### 3.2.1 Training Procedure
Conducted in two phases:

**Phase 1: Model Training**
```bash
python C:\Users\user\Desktop\int sys\option_c\project_option_c\train_p1.py
```

**Training Settings:**
- **Data Retrieval:** yfinance for META (5 years)
- **Features:** Multiple attributes
- **Preprocessing:** MinMaxScaler per attribute
- **Sequence Length:** 50 days
- **Forecast Horizon:** 15 days
- **Batch Size:** 64
- **Epochs:** 50
- **Optimizer:** Adam
- **Loss Function:** Huber
- **Validation Split:** 20%

**Training Output:**
```
Epoch 1/50: loss: 0.0330 - val_loss: 0.0022
Epoch 7/50: val_loss: 0.00198
...
Epoch 49/50: val_loss: 0.00151
```

**Model Storage:**
- **Weights Saving:** Best weights saved
- **Full Model:** Stored as .keras
- **Data Storage:** Saved with joblib
- **Logging:** Metrics tracked in TensorBoard

#### 3.2.2 Evaluation Procedure
**Phase 2: Model Assessment**
```bash
python C:\Users\user\Desktop\int sys\option_c\project_option_c\test_p1.py
```

**Evaluation Workflow:**
1. **Model Load:** Retrieved saved model
2. **Data Prep:** Processed with saved scalers
3. **Prediction:** Inferred on test data
4. **Metric Calculation:** Detailed analysis
5. **Profit Assessment:** Buy/sell profit evaluation
6. **Visualization:** Actual vs predicted plot
7. **Results Export:** Saved to CSV

**Evaluation Features:**
- **Fallback:** Weight load if needed
- **Metrics:** Loss, MAE, accuracy, profit
- **Future Forecast:** 15-day prediction
- **Validation:** Performance check

### 3.3 Training and Evaluation Infrastructure

#### 3.3.1 Directory Structure Created
- **v0.1 Model:** Basic output setup
- **P1 Model:** Organized as:
  ```
  C:\Users\user\Desktop\int sys\option_c\project_option_c\p1\
  ├── data/
  ├── logs/
  ├── results/
  ├── model_checkpoints/
  └── csv-results/
  ```

#### 3.3.2 Hardware and Performance
- **CPU Optimization:** TensorFlow utilized SSE3, SSE4.1, SSE4.2, AVX, AVX2, AVX_VNNI, FMA
- **Memory Management:** Batch-based
- **Training Time:**
  - v0.1 Model: ~35 seconds
  - P1 Model: ~90 seconds

## 4. Performance Outcomes

### 4.1 v0.1 Model Outcomes
**Training Outcomes:**
- **Epochs:** 25
- **Final Loss:** 0.0048
- **Time Taken:** 35 seconds
- **Next Day Prediction:** $111.83

**Observations:**
- Simple design
- Limited to one feature
- Lacks detailed metrics
- Basic visual output

### 4.2 P1 Model Outcomes
**Training Outcomes:**
- **Epochs:** 50
- **Final Loss:** 0.0016 (Huber)
- **Validation Loss:** 0.0016
- **Mean Absolute Error:** 0.0423
- **Time Taken:** 90 seconds

**Testing Outcomes:**
- **Future Prediction (15 days):** $756.37
- **Loss:** 0.00155
- **Mean Absolute Error:** $116.43
- **Accuracy:** 57.74%
- **Buy Profit:** $1,153.20
- **Sell Profit:** $283.77
- **Total Profit:** $1,436.97
- **Profit per Trade:** $6.01

## 5. Code Development Analysis

### 5.1 Key Enhancements from Original to Current

#### 5.1.1 Data Source Transition
**Original:** Relied on `yahoo_fin`
```python
from yahoo_fin import stock_info as si
df = si.get_data(ticker)
```

**Current:** Adopted `yfinance`
```python
import yfinance as yf
ticker_data = yf.download(ticker, period="5y")
```

#### 5.1.2 Improved Data Handling
- Enhanced MultiIndex management
- Standardized naming
- Reliable adjclose fallback
- Stronger error checks

#### 5.1.3 Model Storage and Control
**Original:** Only saved weights
```python
checkpointer = ModelCheckpoint(os.path.join("results", model_name + ".h5"), save_weights_only=True, save_best_only=True, verbose=1)
```

**Current:** Full model and data storage
```python
model_checkpoint_path = os.path.join("C:\Users\user\Desktop\int sys\option_c\project_option_c\p1\model_checkpoints", model_name + ".keras")
save_model(model, model_checkpoint_path)
data_path = os.path.join("C:\Users\user\Desktop\int sys\option_c\project_option_c\p1\model_checkpoints", model_name + "_data.pkl")
joblib.dump(data, data_path)
```

#### 5.1.4 Configuration Management
**Original:** Scattered hardcoded values
**Current:** Centralized in `parameters.py` with flexibility

#### 5.1.5 Updated Loss Function
**Original:** Deprecated `huber_loss`
```python
LOSS = "huber_loss"
```

**Current:** Current `huber`
```python
LOSS = "huber"
```

#### 5.1.6 Enhanced Directory Layout
**Original:** Flat layout
**Current:** Hierarchical layout
```
C:\Users\user\Desktop\int sys\option_c\project_option_c\p1\
├── data/
├── logs/
├── results/
├── model_checkpoints/
└── csv-results/
```

### 5.2 Structural Improvements

#### 5.2.1 Model Complexity
- **v0.1:** 3 layers x 50 units
- **P1:** 2 layers x 256 units

#### 5.2.2 Feature Development
- **v0.1:** Single feature
- **P1:** Multiple features

#### 5.2.3 Evaluation Metrics
- **v0.1:** Minimal output
- **P1:** Extensive metrics including profit

## 6. Comparative Analysis

### 6.1 Model Performance Comparison

| Metric               | v0.1 Model            | P1 Model              | Winner |
|-----------------------|-----------------------|-----------------------|--------|
| Architecture Complexity | Basic (3x50 units)   | Advanced (2x256 units)| P1     |
| Features Used         | 1 (close price)      | 5 (multi-feature)     | P1     |
| Training Epochs       | 25                   | 50                    | P1     |
| Loss Function         | MSE                  | Huber                 | P1     |
| Model Persistence     | Weights only         | Complete model + data | P1     |
| Evaluation Metrics    | Basic                | Comprehensive         | P1     |
| Profit Analysis       | None                 | Detailed ($1,436.97 total) | P1 |
| Code Organization     | Monolithic           | Modular               | P1     |

### 6.2 Technical Enhancements

1. **Data Stability:** Shift to yfinance
2. **Model Durability:** Better validation
3. **Scalability:** Flexible design
4. **Maintainability:** Structured code
5. **Monitoring:** Detailed logs

## 7. Key Observations and Suggestions

### 7.1 Performance Insights
1. **P1 Advantage:** Superior metrics with P1
2. **Multi-feature Gain:** Enhanced accuracy
3. **Design Optimization:** 2x256 beats 3x50

### 7.2 Code Quality Gains
1. **Modularity:** Improved practices
2. **Maintainability:** Unified settings
3. **Extensibility:** Easy updates

### 7.3 Future Improvements
1. **Parameter Tuning:** Refine settings
2. **Advanced Designs:** Test bidirectional LSTMs
3. **Feature Expansion:** Include indicators
4. **Ensemble Approaches:** Mix models

## 8. Conclusion

The shift from v0.1 to P1 reflects significant progress in model effectiveness and code structure. P1 delivers higher accuracy (57.74%, $1,436.97 profit) with a scalable framework. The adoption of yfinance, robust storage, and modern techniques ensures long-term viability. This project showcases LSTM utility and ML engineering standards.

---

**Report Generated:** August 24, 2025, 02:20 PM AEST  
**Total Execution Time:** ~3 minutes  
**Models Successfully Trained and Tested:** 2/2
```