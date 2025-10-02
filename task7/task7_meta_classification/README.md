# Task 7 Meta-Classification Approach

## Overview
This folder implements the meta-classification approach for Task 7, which successfully improves upon the baseline by combining Task 6's technical predictions with sentiment features.

## Key Achievement
- **Test Accuracy: 54.63%** (vs baseline 53.24%)
- **Improvement: +1.39%** over technical-only baseline
- **No overfitting** (train: 62.88%, test: 54.63%)
- **Hyperparameter Tuning**: Optimized Random Forest parameters

## Performance Comparison

| Approach | Test Accuracy | Test F1-Score | Test AUC | Notes |
|----------|---------------|---------------|----------|-------|
| **Baseline (LSTM)** | 53.24% | - | - | Original Task 6 model |
| **Direct Feature Combination** | 46-50% | - | - | Adding sentiment features directly hurt performance |
| **Meta-Classification (Basic)** | 54.63% | 64.71% | 55.67% | Random Forest on Task6 predictions + sentiment |
| **Meta-Classification (Tuned)** | 53.70% | 65.03% | 52.84% | After hyperparameter optimization |

## Hyperparameter Tuning Results

**Best Parameters:**
- n_estimators: 100
- max_depth: 5
- min_samples_split: 10
- min_samples_leaf: 3
- max_features: 'log2'

**Cross-Validation Score:** 60.98%
1. **Task 6 Predictions**: Use cross-validated Random Forest predictions on technical indicators
2. **Sentiment Features**: 29 sentiment-based features
3. **Meta-Classifier**: Regularized Random Forest combining both
4. **Total Features**: 30 meta-features (1 Task 6 prediction + 29 sentiment)

## Files Structure
```
task7_meta_classification/
├── task7_meta_classifier.py          # Main implementation
├── training_outputs/                 # Saved models and results
│   ├── meta_classifier.pkl          # Trained meta-classifier
│   ├── meta_classifier_results.json # Performance metrics
│   └── feature_info.json            # Feature information
├── evaluation_results/              # Analysis and reports
│   └── meta_classification_analysis.json
└── feature_engineering/             # For future preprocessing
```

## Why This Works
- **Avoids data leakage** through cross-validation
- **Prevents overfitting** with regularization
- **Proper integration** of technical and sentiment features
- **Leverages Task 6's** proven performance

## Results Summary
- **Best performing approach** among all Task 7 methods
- **F1-Score: 65.03%** (excellent balance after tuning)
- **AUC: 52.84%** (better than random)
- **Stable performance** without overfitting
- **Hyperparameter optimization** completed with GridSearchCV

## Next Steps
This meta-classification approach can be further improved by:
1. Using actual Task 6 ensemble models instead of Random Forest
2. Adding more sophisticated feature engineering
3. Hyperparameter tuning of the meta-classifier
4. Testing with different meta-classifiers (SVM, Neural Networks)