"""
Task 7 Data Integration Script
Combines META stock price data from Task 6 with daily sentiment features for sentiment-enhanced stock price prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Task7DataIntegrator:
    """Integrates Task 6 stock data with Task 7 sentiment features."""

    def __init__(self, task6_data_dir: str, task7_data_dir: str, output_dir: str):
        """
        Initialize the data integrator.

        Args:
            task6_data_dir: Path to Task 6 data directory
            task7_data_dir: Path to Task 7 data directory
            output_dir: Path to output directory for integrated data
        """
        self.task6_data_dir = Path(task6_data_dir)
        self.task7_data_dir = Path(task7_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_stock_data(self) -> pd.DataFrame:
        """Load and combine META stock data from Task 6."""
        logger.info("Loading META stock data from Task 6...")

        # Load both stock data files
        file1 = self.task6_data_dir / "META_2020-01-01_2023-08-01.csv"
        file2 = self.task6_data_dir / "META_2023-08-02_2024-07-02.csv"

        # Read and combine the data - skip the first 4 rows (headers)
        df1 = pd.read_csv(file1, skiprows=4, header=None,
                         names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
        df2 = pd.read_csv(file2, skiprows=4, header=None,
                         names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])

        # Combine the dataframes
        stock_df = pd.concat([df1, df2], ignore_index=True)

        # Convert Date column to datetime
        stock_df['Date'] = pd.to_datetime(stock_df['Date'])

        # Sort by date
        stock_df = stock_df.sort_values('Date').reset_index(drop=True)

        # Rename columns for clarity
        stock_df = stock_df.rename(columns={
            'Close': 'close',
            'High': 'high',
            'Low': 'low',
            'Open': 'open',
            'Volume': 'volume'
        })

        logger.info(f"Loaded {len(stock_df)} stock price records from {stock_df['Date'].min()} to {stock_df['Date'].max()}")
        return stock_df

    def load_sentiment_data(self) -> pd.DataFrame:
        """Load daily sentiment features from Task 7."""
        logger.info("Loading daily sentiment features from Task 7...")

        sentiment_file = self.task7_data_dir / "data_preprocessing" / "daily_sentiment_features.csv"
        sentiment_df = pd.read_csv(sentiment_file)

        # Convert date column to datetime
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

        logger.info(f"Loaded {len(sentiment_df)} daily sentiment records from {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
        return sentiment_df

    def create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators similar to Task 6."""
        logger.info("Creating technical indicators...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()

        # Exponential moving averages
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # RSI (Relative Strength Index)
        def calculate_rsi(data, window=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))

        df['rsi_14'] = calculate_rsi(df['close'], 14)

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # Volume indicators
        df['volume_sma_5'] = df['volume'].rolling(window=5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_5']

        # Volatility (standard deviation of returns)
        df['volatility_5'] = df['returns'].rolling(window=5).std()
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        # Price momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1

        # Target variable: next day's return (for prediction)
        df['target_return'] = df['returns'].shift(-1)
        df['target_direction'] = (df['target_return'] > 0).astype(int)  # 1 for up, 0 for down

        logger.info("Technical indicators created successfully")
        return df

    def merge_data(self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock data with sentiment features."""
        logger.info("Merging stock data with sentiment features...")

        # Merge on date (stock data uses 'Date', sentiment uses 'date')
        merged_df = pd.merge(stock_df, sentiment_df,
                           left_on='Date', right_on='date',
                           how='left')

        # Drop the duplicate date column
        merged_df = merged_df.drop('date', axis=1)

        # Fill missing sentiment data with 0 (neutral sentiment)
        sentiment_cols = [col for col in merged_df.columns if col.startswith(('avg_', 'std_', 'total_', 'sentiment_', 'pos_neg_'))]
        merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)

        # Fill missing article_count with 0
        merged_df['article_count'] = merged_df['article_count'].fillna(0)

        logger.info(f"Merged data has {len(merged_df)} rows with {len(merged_df.columns)} columns")
        logger.info(f"Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
        logger.info(f"Days with sentiment data: {len(merged_df[merged_df['article_count'] > 0])}")
        logger.info(f"Days without sentiment data: {len(merged_df[merged_df['article_count'] == 0])}")

        return merged_df

    def create_lag_features(self, df: pd.DataFrame, lag_days: int = 3) -> pd.DataFrame:
        """Create lagged sentiment features for prediction."""
        logger.info(f"Creating lagged sentiment features ({lag_days} days)...")

        df = df.copy()
        sentiment_cols = [col for col in df.columns if col.startswith(('avg_', 'std_', 'total_', 'sentiment_', 'pos_neg_'))]

        # Create lagged versions of sentiment features
        for col in sentiment_cols:
            for lag in range(1, lag_days + 1):
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

        logger.info(f"Created {len(sentiment_cols) * lag_days} lagged sentiment features")
        return df

    def save_integrated_data(self, df: pd.DataFrame, filename: str = "task7_integrated_data.csv"):
        """Save the integrated dataset."""
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Integrated data saved to {output_path}")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Save data info
        info_path = self.output_dir / "data_integration_info.txt"
        with open(info_path, 'w') as f:
            f.write("Task 7 Data Integration Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Date range: {df['Date'].min()} to {df['Date'].max()}\n")
            f.write(f"Columns: {len(df.columns)}\n\n")

            f.write("Stock price columns:\n")
            stock_cols = ['Date', 'open', 'high', 'low', 'close', 'volume']
            for col in stock_cols:
                f.write(f"  - {col}\n")

            f.write("\nTechnical indicator columns:\n")
            tech_cols = [col for col in df.columns if any(x in col for x in ['sma_', 'ema_', 'macd', 'rsi_', 'bb_', 'volatility_', 'momentum_'])]
            for col in tech_cols[:10]:  # Show first 10
                f.write(f"  - {col}\n")
            if len(tech_cols) > 10:
                f.write(f"  ... and {len(tech_cols) - 10} more\n")

            f.write("\nSentiment feature columns:\n")
            sentiment_cols = [col for col in df.columns if col.startswith(('avg_', 'std_', 'total_', 'sentiment_', 'pos_neg_', 'article_count'))]
            for col in sentiment_cols:
                f.write(f"  - {col}\n")

            f.write("\nLagged feature columns:\n")
            lagged_cols = [col for col in df.columns if '_lag_' in col]
            f.write(f"  - {len(lagged_cols)} lagged sentiment features\n")

            f.write("\nTarget columns:\n")
            target_cols = [col for col in df.columns if col.startswith('target_')]
            for col in target_cols:
                f.write(f"  - {col}\n")

    def integrate_data(self) -> pd.DataFrame:
        """Main integration pipeline."""
        logger.info("Starting Task 7 data integration...")

        # Load data
        stock_df = self.load_stock_data()
        sentiment_df = self.load_sentiment_data()

        # Create technical indicators
        stock_df = self.create_technical_indicators(stock_df)

        # Merge data
        integrated_df = self.merge_data(stock_df, sentiment_df)

        # Create lagged features
        integrated_df = self.create_lag_features(integrated_df)

        # Remove rows with NaN values (due to technical indicators and lagging)
        initial_rows = len(integrated_df)
        integrated_df = integrated_df.dropna()
        final_rows = len(integrated_df)

        logger.info(f"Removed {initial_rows - final_rows} rows with NaN values")
        logger.info(f"Final dataset: {final_rows} rows, {len(integrated_df.columns)} columns")

        # Save integrated data
        self.save_integrated_data(integrated_df)

        return integrated_df

def main():
    """Main execution function."""
    # Define paths
    task6_data_dir = "d:/project_option_C/project_option_c/task6/data"
    task7_data_dir = "d:/project_option_C/project_option_c/task7"
    output_dir = "d:/project_option_C/project_option_c/task7/data_integration"

    # Create integrator and run integration
    integrator = Task7DataIntegrator(task6_data_dir, task7_data_dir, output_dir)
    integrated_data = integrator.integrate_data()

    logger.info("Task 7 data integration completed successfully!")
    logger.info(f"Integrated dataset shape: {integrated_data.shape}")
    logger.info(f"Date range: {integrated_data['Date'].min()} to {integrated_data['Date'].max()}")

if __name__ == "__main__":
    main()