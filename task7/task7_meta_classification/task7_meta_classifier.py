"""
Task 7 Meta-Classification Approach
Uses Task 6's ensemble predictions as features for sentiment-enhanced classification.

This approach:
1. Gets predictions from Task 6's ensemble model
2. Combines them with sentiment features
3. Trains a meta-classifier for final predictions
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import from Task 6
import sys
import os
task6_path = os.path.join(os.path.dirname(__file__), '..', '..', 'task6')
sys.path.insert(0, task6_path)

try:
    from ensemble_methods import ensemble_average
    from model_factory import build_model
    TASK6_AVAILABLE = True
    print("✅ Task 6 modules imported successfully")
except ImportError as e:
    TASK6_AVAILABLE = False
    print(f"⚠️  Task 6 modules not available: {e}")

# Import from Task 7 data processing
task7_data_path = os.path.join(os.path.dirname(__file__), '..', 'train_data_split')
sys.path.insert(0, task7_data_path)

class Task7MetaClassifier:
    def __init__(self, output_dir="training_outputs"):
        """Initialize the meta-classifier"""
        self.output_dir = output_dir
        self.task6_predictions = None
        self.sentiment_features = None
        self.meta_features = None
        self.meta_classifier = None
        self.results = {}

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    def load_task7_data(self):
        """Load Task 7's sentiment-enhanced data"""
        print("Loading Task 7 sentiment-enhanced data...")

        try:
            # Load the split data from Task 7
            train_path = os.path.join(os.path.dirname(__file__), '..', 'train_data_split', 'train_data.csv')
            test_path = os.path.join(os.path.dirname(__file__), '..', 'train_data_split', 'test_data.csv')

            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)

            print(f"Loaded train set: {len(self.train_data)} samples")
            print(f"Loaded test set: {len(self.test_data)} samples")

            # Separate features and target
            feature_cols = [col for col in self.train_data.columns if col != 'target_direction']
            self.X_train = self.train_data[feature_cols]
            self.y_train = self.train_data['target_direction']
            self.X_test = self.test_data[feature_cols]
            self.y_test = self.test_data['target_direction']

            print(f"Features: {len(feature_cols)} total features")

            return True
        except Exception as e:
            print(f"Error loading Task 7 data: {e}")
            return False

    def get_task6_predictions(self):
        """Get Task 6 predictions using cross-validation to avoid data leakage"""
        print("Getting Task 6 predictions using cross-validation...")

        from sklearn.model_selection import KFold
        from sklearn.ensemble import RandomForestClassifier

        # Get technical features only
        technical_features = [col for col in self.X_train.columns
                            if not col.startswith(('sentiment_', 'total_', 'pos_neg_', 'std_')) and col != 'Date']

        print(f"Using {len(technical_features)} technical features for Task 6 predictions")

        # Use 5-fold cross-validation to get unbiased predictions for training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        # Initialize arrays to store predictions
        self.task6_train_pred = np.zeros(len(self.X_train))

        # For each fold, train on 4/5 of data, predict on 1/5 (unseen data)
        for train_idx, val_idx in kf.split(self.X_train):
            # Train Task 6 model on training portion
            task6_model = RandomForestClassifier(n_estimators=100, random_state=42)
            task6_model.fit(self.X_train.iloc[train_idx][technical_features],
                          self.y_train.iloc[train_idx])

            # Predict on validation portion (unseen data for this fold)
            fold_pred = task6_model.predict_proba(self.X_train.iloc[val_idx][technical_features])[:, 1]
            self.task6_train_pred[val_idx] = fold_pred

        # For test set, train on full training data and predict
        task6_full_model = RandomForestClassifier(n_estimators=100, random_state=42)
        task6_full_model.fit(self.X_train[technical_features], self.y_train)
        self.task6_test_pred = task6_full_model.predict_proba(self.X_test[technical_features])[:, 1]

        print("✅ Task 6 predictions generated using cross-validation (no data leakage)")
        return True

    def create_meta_features(self):
        """Create meta-features by combining Task 6 predictions with sentiment features"""
        print("Creating meta-features...")

        # Get sentiment feature columns
        sentiment_features = [col for col in self.X_train.columns
                            if col.startswith(('sentiment_', 'total_', 'pos_neg_', 'std_'))]

        print(f"Using {len(sentiment_features)} sentiment features")

        # Create meta-features
        # Feature 1: Task 6 ensemble prediction
        # Features 2+: Sentiment features

        self.meta_train = np.column_stack([
            self.task6_train_pred.reshape(-1, 1),  # Task 6 prediction as feature
            self.X_train[sentiment_features].values  # Sentiment features
        ])

        self.meta_test = np.column_stack([
            self.task6_test_pred.reshape(-1, 1),  # Task 6 prediction as feature
            self.X_test[sentiment_features].values  # Sentiment features
        ])

        print(f"Meta-features created: Train {self.meta_train.shape}, Test {self.meta_test.shape}")
        print(f"Features: Task6_Prediction + {len(sentiment_features)} sentiment features")

        return True

    def train_meta_classifier(self):
        """Train the meta-classifier with hyperparameter tuning"""
        print("Training meta-classifier with hyperparameter tuning...")

        from sklearn.model_selection import GridSearchCV

        # Define hyperparameter grid for Random Forest (optimized for speed)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 7, 10],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [3, 5],
            'max_features': ['sqrt', 'log2']
        }

        # Create base model
        base_model = RandomForestClassifier(random_state=42)

        # Perform grid search with cross-validation
        print("Performing hyperparameter tuning with 3-fold CV...")
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        # Fit the grid search
        grid_search.fit(self.meta_train, self.y_train)

        # Get best model
        self.meta_classifier = grid_search.best_estimator_

        # Store grid search results for saving
        self.best_params = grid_search.best_params_
        self.cv_best_score = grid_search.best_score_

        print("✅ Meta-classifier trained with hyperparameter tuning")
        return True

    def evaluate_meta_classifier(self):
        """Evaluate the meta-classifier performance"""
        print("Evaluating meta-classifier...")

        # Get predictions
        train_pred = self.meta_classifier.predict(self.meta_train)
        test_pred = self.meta_classifier.predict(self.meta_test)

        train_pred_proba = self.meta_classifier.predict_proba(self.meta_train)[:, 1]
        test_pred_proba = self.meta_classifier.predict_proba(self.meta_test)[:, 1]

        # Calculate metrics
        results = {
            'train': {
                'accuracy': accuracy_score(self.y_train, train_pred),
                'precision': precision_score(self.y_train, train_pred),
                'recall': recall_score(self.y_train, train_pred),
                'f1_score': f1_score(self.y_train, train_pred),
                'auc': roc_auc_score(self.y_train, train_pred_proba)
            },
            'test': {
                'accuracy': accuracy_score(self.y_test, test_pred),
                'precision': precision_score(self.y_test, test_pred),
                'recall': recall_score(self.y_test, test_pred),
                'f1_score': f1_score(self.y_test, test_pred),
                'auc': roc_auc_score(self.y_test, test_pred_proba)
            }
        }

        self.results = results

        # Print results
        print("\n" + "="*60)
        print("META-CLASSIFIER RESULTS")
        print("="*60)
        print(f"Train Accuracy: {results['train']['accuracy']:.4f}")
        print(f"Test Accuracy:  {results['test']['accuracy']:.4f}")
        print(f"Train F1-Score: {results['train']['f1_score']:.4f}")
        print(f"Test F1-Score:  {results['test']['f1_score']:.4f}")
        print(f"Train AUC:      {results['train']['auc']:.4f}")
        print(f"Test AUC:       {results['test']['auc']:.4f}")
        print("="*60)

        return results

    def save_results(self):
        """Save all results and models"""
        print(f"Saving results to {self.output_dir}...")

        # Save model
        model_path = os.path.join(self.output_dir, 'meta_classifier.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.meta_classifier, f)

        # Save results
        results_path = os.path.join(self.output_dir, 'meta_classifier_results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        # Save feature information
        feature_info = {
            'task6_prediction_feature': True,
            'sentiment_features_count': self.meta_train.shape[1] - 1,
            'total_meta_features': self.meta_train.shape[1],
            'best_hyperparameters': self.best_params,
            'cv_best_score': self.cv_best_score
        }

        feature_path = os.path.join(self.output_dir, 'feature_info.json')
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)

        print("✅ Results saved")
        return True

    def run_meta_classification(self):
        """Run the complete meta-classification pipeline"""
        print("="*60)
        print("TASK 7 META-CLASSIFICATION PIPELINE")
        print("="*60)

        # Step 1: Load Task 7 data
        if not self.load_task7_data():
            return False

        # Step 2: Get Task 6 predictions
        if not self.get_task6_predictions():
            return False

        # Step 3: Create meta-features
        if not self.create_meta_features():
            return False

        # Step 4: Train meta-classifier
        if not self.train_meta_classifier():
            return False

        # Step 5: Evaluate
        results = self.evaluate_meta_classifier()

        # Step 6: Save results
        self.save_results()

        print("\n🎉 Meta-classification completed!")
        print(f"Output directory: {self.output_dir}")

        return results


def main():
    """Main function to run meta-classification"""
    # Create meta-classifier
    meta_classifier = Task7MetaClassifier()

    # Run the pipeline
    results = meta_classifier.run_meta_classification()

    if results:
        print("\n📊 FINAL RESULTS SUMMARY:")
        print(f"Train Accuracy: {results['train']['accuracy']:.2f}")
        print(f"Test Accuracy:  {results['test']['accuracy']:.2f}")
        print(f"Train F1-Score: {results['train']['f1_score']:.2f}")
        print(f"Test F1-Score:  {results['test']['f1_score']:.2f}")
        print(f"Train AUC:      {results['train']['auc']:.2f}")
        print(f"Test AUC:       {results['test']['auc']:.2f}")
if __name__ == "__main__":
    main()