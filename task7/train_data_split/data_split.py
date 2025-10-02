"""
Task 7: Data Splitting Script
Split the feature engineered data into train/test sets for model training.

This script:
1. Loads the feature engineered dataset
2. Loads selected features from feature engineering
3. Performs chronological train/test split (80/20)
4. Saves split datasets for model training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import json

class Task7DataSplitter:
    def __init__(self):
        """Initialize the data splitter"""
        self.scaler = StandardScaler()

    def load_data(self):
        """Load the feature engineered dataset and selected features"""
        print("Loading feature engineered dataset...")

        # Paths
        data_path = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\task7_feature_engineered.csv'
        selected_features_path = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\selected_features.json'
        output_dir = 'd:\\project_option_C\\project_option_c\\task7\\train_data_split'

        try:
            # Load data
            self.df = pd.read_csv(data_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])

            # Load selected features
            with open(selected_features_path, 'r') as f:
                selected_data = json.load(f)
            self.selected_features = selected_data['selected_features']

            # Use only selected features
            self.feature_cols = self.selected_features
            self.X = self.df[self.feature_cols]
            self.y = self.df['target_direction']

            print(f"Loaded {len(self.df)} samples with {len(self.feature_cols)} selected features")
            print(f"Date range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f"Class distribution: {self.y.value_counts().to_dict()}")

            self.output_dir = output_dir
            return True

        except FileNotFoundError as e:
            print(f"Error: Could not find file - {e}")
            return False

    def analyze_features(self):
        """Analyze the composition of selected features"""
        print("\nAnalyzing feature composition...")

        # Separate features by type
        technical_features = [col for col in self.feature_cols if not any(sentiment_word in col.lower() for sentiment_word in
                              ['polarity', 'sentiment', 'positive', 'negative', 'article', 'textblob', 'custom'])]
        sentiment_features = [col for col in self.feature_cols if any(sentiment_word in col.lower() for sentiment_word in
                             ['polarity', 'sentiment', 'positive', 'negative', 'article', 'textblob', 'custom'])]

        print(f"Technical features: {len(technical_features)}")
        print(f"Sentiment features: {len(sentiment_features)}")
        print(f"Total selected features: {len(self.feature_cols)}")

        # Save feature analysis
        feature_analysis = {
            'total_features': len(self.feature_cols),
            'technical_features': len(technical_features),
            'sentiment_features': len(sentiment_features),
            'technical_feature_list': technical_features,
            'sentiment_feature_list': sentiment_features
        }

        with open(f'{self.output_dir}\\feature_analysis.json', 'w') as f:
            json.dump(feature_analysis, f, indent=2)

        return technical_features, sentiment_features

    def split_data(self, test_size=0.2):
        """Split data into train/test sets chronologically"""
        print(f"\nSplitting data with test_size={test_size}...")

        # Chronological split (time series)
        split_idx = int(len(self.df) * (1 - test_size))

        # Split data
        train_df = self.df.iloc[:split_idx].copy()
        test_df = self.df.iloc[split_idx:].copy()

        print(f"Train set: {len(train_df)} samples ({train_df['Date'].min()} to {train_df['Date'].max()})")
        print(f"Test set: {len(test_df)} samples ({test_df['Date'].min()} to {test_df['Date'].max()})")

        # Separate features and targets
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_direction']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target_direction']

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Create scaled dataframes
        train_scaled_df = pd.DataFrame(X_train_scaled, columns=self.feature_cols)
        train_scaled_df['target_direction'] = y_train.values
        train_scaled_df['Date'] = train_df['Date'].values

        test_scaled_df = pd.DataFrame(X_test_scaled, columns=self.feature_cols)
        test_scaled_df['target_direction'] = y_test.values
        test_scaled_df['Date'] = test_df['Date'].values

        # Save split data
        train_scaled_df.to_csv(f'{self.output_dir}\\train_data.csv', index=False)
        test_scaled_df.to_csv(f'{self.output_dir}\\test_data.csv', index=False)

        # Save scaler
        import pickle
        with open(f'{self.output_dir}\\scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

        print(f"Split data saved to {self.output_dir}")

        # Print class distributions
        print("\nTrain set class distribution:")
        print(train_df['target_direction'].value_counts())
        print("\nTest set class distribution:")
        print(test_df['target_direction'].value_counts())

        return train_scaled_df, test_scaled_df

    def create_data_summary(self):
        """Create a summary of the split data"""
        print("\nCreating data summary...")

        summary = {
            'total_samples': len(self.df),
            'train_samples': int(len(self.df) * 0.8),
            'test_samples': int(len(self.df) * 0.2),
            'features_count': len(self.feature_cols),
            'date_range': {
                'start': str(self.df['Date'].min()),
                'end': str(self.df['Date'].max())
            },
            'class_distribution': self.y.value_counts().to_dict(),
            'split_method': 'chronological_80_20'
        }

        with open(f'{self.output_dir}\\data_split_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print("Data summary saved")

    def run_pipeline(self):
        """Run the complete data splitting pipeline"""
        print("=== Task 7 Data Splitting Pipeline ===\n")

        # Load data
        if not self.load_data():
            return False

        # Analyze features
        self.analyze_features()

        # Split data
        self.split_data()

        # Create summary
        self.create_data_summary()

        print("\n=== Data Splitting Pipeline Complete ===")
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print("- train_data.csv: Training dataset")
        print("- test_data.csv: Testing dataset")
        print("- scaler.pkl: Feature scaler")
        print("- feature_analysis.json: Feature composition analysis")
        print("- data_split_summary.json: Data split summary")

        return True

def main():
    """Main function to run the data splitting pipeline"""
    splitter = Task7DataSplitter()
    success = splitter.run_pipeline()

    if success:
        print("\nData splitting completed successfully!")
        print("Next step: Use the split data for model training")
    else:
        print("Data splitting failed. Please check data paths and dependencies.")

if __name__ == "__main__":
    main()