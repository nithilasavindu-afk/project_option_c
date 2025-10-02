"""
Task 7 Feature Engineering Pipeline
Advanced feature engineering for sentiment-enhanced stock price prediction.

This script handles:
1. Data loading and preprocessing
2. Feature scaling and normalization
3. Interaction feature creation
4. Feature selection and importance analysis
5. Data preparation for ensemble modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class Task7FeatureEngineer:
    def __init__(self):
        """Initialize the feature engineer"""
        self.scalers = {}
        self.feature_importance = {}
        self.selected_features = []

    def load_data(self):
        """Load the integrated dataset"""
        print("Loading integrated dataset...")
        data_path = 'd:\\project_option_C\\project_option_c\\task7\\data_integration\\task7_integrated_data.csv'

        try:
            self.df = pd.read_csv(data_path)
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            self.df = self.df.sort_values('Date').reset_index(drop=True)

            print(f"Loaded {len(self.df)} records from {self.df['Date'].min()} to {self.df['Date'].max()}")
            print(f"Shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {data_path}")
            return False

    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        print("Handling missing values...")

        # Check for missing values
        missing_info = self.df.isnull().sum()
        missing_cols = missing_info[missing_info > 0]

        if len(missing_cols) > 0:
            print(f"Found missing values in {len(missing_cols)} columns:")
            for col, count in missing_cols.items():
                print(f"  {col}: {count} missing values")

            # Forward fill for time series data
            self.df = self.df.fillna(method='ffill')

            # For any remaining NaN at the beginning, use backward fill
            self.df = self.df.fillna(method='bfill')

            print("Missing values handled using forward/backward fill")
        else:
            print("No missing values found")

    def create_technical_features(self):
        """Create advanced technical indicator features"""
        print("Creating advanced technical features...")

        df = self.df.copy()

        # Trend strength indicators
        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['momentum_acceleration'] = df['momentum_5'] - df['momentum_5'].shift(1)

        # Volatility ratios
        df['volatility_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volume indicators
        df['volume_trend'] = (df['volume'] - df['volume_sma_5']) / df['volume_sma_5']
        df['volume_price_trend'] = df['volume_trend'] * df['returns']

        # MACD signals
        df['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_divergence'] = df['macd'] - df['macd_signal']

        # RSI signals
        df['rsi_overbought'] = np.where(df['rsi_14'] > 70, 1, 0)
        df['rsi_oversold'] = np.where(df['rsi_14'] < 30, 1, 0)

        self.df = df
        print(f"Created {len(df.columns) - len(self.df.columns)} additional technical features")

    def create_sentiment_features(self):
        """Create advanced sentiment-based features"""
        print("Creating advanced sentiment features...")

        df = self.df.copy()

        # Sentiment momentum
        df['sentiment_momentum'] = df['avg_textblob_polarity'] - df['avg_textblob_polarity'].shift(1)
        df['sentiment_acceleration'] = df['sentiment_momentum'] - df['sentiment_momentum'].shift(1)

        # Sentiment volatility measures
        df['sentiment_range'] = df['avg_textblob_polarity'].rolling(5).max() - df['avg_textblob_polarity'].rolling(5).min()
        df['sentiment_trend'] = df['avg_textblob_polarity'].rolling(5).mean()

        # News volume impact
        df['news_intensity'] = df['article_count'] * df['sentiment_intensity']
        df['sentiment_weighted_volume'] = df['avg_textblob_polarity'] * df['article_count']

        # Sentiment-price divergence
        df['sentiment_price_divergence'] = df['avg_textblob_polarity'] - df['returns'].shift(1)

        # Custom sentiment indicators
        df['custom_sentiment_trend'] = df['avg_custom_polarity'].rolling(3).mean()
        df['sentiment_consistency'] = 1 - df['std_textblob_polarity']

        self.df = df
        print(f"Created {len(df.columns) - len(self.df.columns)} additional sentiment features")

    def create_interaction_features(self):
        """Create interaction features between sentiment and technical indicators"""
        print("Creating interaction features...")

        df = self.df.copy()

        # Sentiment-technical interactions
        df['sentiment_rsi_interaction'] = df['avg_textblob_polarity'] * df['rsi_14']
        df['sentiment_volatility_interaction'] = df['avg_textblob_polarity'] * df['volatility_20']
        df['sentiment_momentum_interaction'] = df['avg_textblob_polarity'] * df['momentum_5']

        # News volume-technical interactions
        df['news_volume_price_interaction'] = df['article_count'] * df['returns']
        df['news_sentiment_macd'] = df['avg_textblob_polarity'] * df['macd']

        # Volatility-sentiment interactions
        df['high_volatility_sentiment'] = np.where(df['volatility_20'] > df['volatility_20'].quantile(0.75),
                                                 df['avg_textblob_polarity'], 0)
        df['low_volatility_sentiment'] = np.where(df['volatility_20'] < df['volatility_20'].quantile(0.25),
                                                df['avg_textblob_polarity'], 0)

        # Trend-sentiment alignment
        df['bullish_sentiment_trend'] = np.where(df['trend_strength'] > 0, df['avg_textblob_polarity'], 0)
        df['bearish_sentiment_trend'] = np.where(df['trend_strength'] < 0, df['avg_textblob_polarity'], 0)

        self.df = df
        print(f"Created {len(df.columns) - len(self.df.columns)} interaction features")

    def scale_features(self):
        """Scale features using appropriate scalers"""
        print("Scaling features...")

        # Separate features by type for appropriate scaling
        price_features = ['close', 'high', 'low', 'open', 'volume']
        technical_features = [col for col in self.df.columns if any(x in col.lower() for x in
                          ['sma', 'ema', 'macd', 'rsi', 'bb', 'volatility', 'momentum'])]
        sentiment_features = [col for col in self.df.columns if any(x in col.lower() for x in
                           ['polarity', 'sentiment', 'positive', 'negative', 'article'])]
        interaction_features = [col for col in self.df.columns if 'interaction' in col.lower()]

        # Use RobustScaler for features with outliers (most financial features)
        robust_features = price_features + technical_features + interaction_features
        if robust_features:
            self.scalers['robust'] = RobustScaler()
            scaled_robust = self.scalers['robust'].fit_transform(self.df[robust_features])
            for i, col in enumerate(robust_features):
                self.df[f'{col}_scaled'] = scaled_robust[:, i]

        # Use StandardScaler for sentiment features
        if sentiment_features:
            self.scalers['standard'] = StandardScaler()
            scaled_standard = self.scalers['standard'].fit_transform(self.df[sentiment_features])
            for i, col in enumerate(sentiment_features):
                self.df[f'{col}_scaled'] = scaled_standard[:, i]

        print(f"Scaled {len(robust_features)} features with RobustScaler and {len(sentiment_features)} with StandardScaler")

    def select_features(self, k=50):
        """Perform feature selection using multiple methods"""
        print(f"Performing feature selection (selecting top {k} features)...")

        # Prepare feature matrix and target
        feature_cols = [col for col in self.df.columns if col not in ['Date', 'target_return', 'target_direction']]
        X = self.df[feature_cols].fillna(0)
        y = self.df['target_return'].fillna(0)

        # Method 1: F-regression (linear relationships)
        selector_f = SelectKBest(score_func=f_regression, k=k)
        X_f_selected = selector_f.fit_transform(X, y)
        f_scores = selector_f.scores_
        f_features = X.columns[selector_f.get_support()].tolist()

        # Method 2: Mutual Information (non-linear relationships)
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=k)
        X_mi_selected = selector_mi.fit_transform(X, y)
        mi_scores = selector_mi.scores_
        mi_features = X.columns[selector_mi.get_support()].tolist()

        # Combine features from both methods
        combined_features = list(set(f_features + mi_features))

        # If we have more than k features, select the best ones
        if len(combined_features) > k:
            # Calculate combined scores
            feature_scores = {}
            for feature in combined_features:
                f_score = f_scores[X.columns.get_loc(feature)] if feature in f_features else 0
                mi_score = mi_scores[X.columns.get_loc(feature)] if feature in mi_features else 0
                feature_scores[feature] = f_score + mi_score

            combined_features = sorted(feature_scores.keys(), key=lambda x: feature_scores[x], reverse=True)[:k]

        self.selected_features = combined_features
        self.feature_importance = {
            'f_regression_scores': dict(zip(X.columns, f_scores)),
            'mutual_info_scores': dict(zip(X.columns, mi_scores)),
            'selected_features': combined_features
        }

        print(f"Selected {len(combined_features)} features using combined F-regression and Mutual Information")

    def create_sequences(self, sequence_length=10):
        """Create sequences for time series modeling"""
        print(f"Creating sequences of length {sequence_length}...")

        sequences = []
        targets = []

        selected_cols = self.selected_features + ['target_return', 'target_direction']

        for i in range(len(self.df) - sequence_length):
            seq_data = self.df[selected_cols].iloc[i:i+sequence_length].values
            target_return = self.df['target_return'].iloc[i+sequence_length]
            target_direction = self.df['target_direction'].iloc[i+sequence_length]

            sequences.append(seq_data)
            targets.append([target_return, target_direction])

        self.sequences = np.array(sequences)
        self.targets = np.array(targets)

        print(f"Created {len(sequences)} sequences with shape {self.sequences.shape}")

    def save_processed_data(self):
        """Save all processed datasets"""
        print("Saving processed data...")

        # Save feature engineered dataframe
        feature_output = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\task7_feature_engineered.csv'
        self.df.to_csv(feature_output, index=False)
        print(f"Saved feature engineered data: {feature_output}")

        # Save selected features
        selected_output = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\selected_features.json'
        import json
        with open(selected_output, 'w') as f:
            json.dump(self.feature_importance, f, indent=2)
        print(f"Saved feature selection results: {selected_output}")

        # Save sequences for deep learning models
        if hasattr(self, 'sequences'):
            seq_output = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\sequences.npz'
            np.savez(seq_output, sequences=self.sequences, targets=self.targets)
            print(f"Saved sequences: {seq_output}")

        # Save scalers
        import pickle
        scaler_output = 'd:\\project_option_C\\project_option_c\\task7\\feature_engineering\\scalers.pkl'
        with open(scaler_output, 'wb') as f:
            pickle.dump(self.scalers, f)
        print(f"Saved scalers: {scaler_output}")

    def run_full_pipeline(self):
        """Run the complete feature engineering pipeline"""
        print("=== Task 7 Feature Engineering Pipeline ===\n")

        # Load data
        if not self.load_data():
            return False

        # Handle missing values
        self.handle_missing_values()

        # Create advanced features
        self.create_technical_features()
        self.create_sentiment_features()
        self.create_interaction_features()

        # Scale features
        self.scale_features()

        # Feature selection
        self.select_features(k=50)

        # Create sequences for time series modeling
        self.create_sequences(sequence_length=10)

        # Save results
        self.save_processed_data()

        print("\n=== Feature Engineering Pipeline Complete ===")
        print(f"Original features: {len(self.df.columns)}")
        print(f"Selected features: {len(self.selected_features)}")
        print(f"Created sequences: {len(self.sequences) if hasattr(self, 'sequences') else 0}")

        return True

def main():
    """Main function to run the feature engineering pipeline"""
    engineer = Task7FeatureEngineer()
    success = engineer.run_full_pipeline()

    if success:
        print("\nNext steps:")
        print("1. Review the feature engineered data in feature_engineering folder")
        print("2. Train sentiment-enhanced ensemble models")
        print("3. Compare performance with Task 6 baseline models")
    else:
        print("Feature engineering failed. Please check data paths and dependencies.")

if __name__ == "__main__":
    main()