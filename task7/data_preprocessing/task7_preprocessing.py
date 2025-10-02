"""
Task 7 Data Preprocessing Pipeline
Preprocesses textual news data for sentiment analysis and feature extraction.

This script handles:
1. Text cleaning and normalization
2. Sentiment analysis and feature extraction
3. Data integration with stock prices
4. Feature engineering for ensemble models
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class Task7DataPreprocessor:
    def __init__(self):
        """Initialize the preprocessor with required NLTK resources"""
        try:
            # Download required NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            print("Warning: Some NLTK resources may not be available")

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Additional domain-specific stop words
        self.domain_stops = {
            'meta', 'facebook', 'platforms', 'company', 'companies', 'inc',
            'corp', 'ltd', 'llc', 'said', 'says', 'told', 'according',
            'reported', 'reportedly', 'announced', 'revealed', 'disclosed'
        }
        self.stop_words.update(self.domain_stops)

    def load_data(self):
        """Load the combined news dataset"""
        print("Loading combined news dataset...")
        data_path = 'd:\\project_option_C\\project_option_c\\task7\\combined_data\\combined_meta_news.csv'

        try:
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df)} articles from {self.df['date'].min()} to {self.df['date'].max()}")
            print(f"Data sources: {self.df['data_source'].value_counts().to_dict()}")
            return True
        except FileNotFoundError:
            print(f"Error: Could not find {data_path}")
            return False

    def clean_text(self, text):
        """Clean and normalize text data"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters and numbers (keep some punctuation for sentiment)
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenize and lemmatize text"""
        try:
            # Tokenize
            tokens = word_tokenize(text)

            # Remove stop words and lemmatize
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    lemma = self.lemmatizer.lemmatize(token)
                    processed_tokens.append(lemma)

            return ' '.join(processed_tokens)
        except:
            return text

    def extract_sentiment_features(self, text):
        """Extract comprehensive sentiment features"""
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Custom sentiment scoring based on financial keywords
        positive_words = {
            'beat', 'beats', 'strong', 'growth', 'profit', 'surge', 'rise',
            'gain', 'bullish', 'positive', 'optimistic', 'record', 'high',
            'increase', 'improvement', 'success', 'win', 'victory', 'boost'
        }

        negative_words = {
            'miss', 'misses', 'weak', 'decline', 'loss', 'plunge', 'fall',
            'drop', 'bearish', 'negative', 'pessimistic', 'low', 'decrease',
            'worse', 'failure', 'concern', 'worry', 'risk', 'threat'
        }

        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)

        # Calculate custom sentiment score
        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            custom_polarity = (positive_count - negative_count) / total_sentiment_words
        else:
            custom_polarity = 0

        # Intensity features
        text_length = len(text)
        word_count = len(words)
        avg_word_length = np.mean([len(word) for word in words]) if words else 0

        return {
            'textblob_polarity': polarity,
            'textblob_subjectivity': subjectivity,
            'custom_polarity': custom_polarity,
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'text_length': text_length,
            'word_count': word_count,
            'avg_word_length': avg_word_length
        }

    def preprocess_text_data(self):
        """Main text preprocessing pipeline"""
        print("Starting text preprocessing...")

        # Clean text
        print("Cleaning text...")
        self.df['cleaned_content'] = self.df['content'].apply(self.clean_text)
        self.df['cleaned_title'] = self.df['title'].apply(self.clean_text)

        # Combine title and content for analysis
        self.df['combined_text'] = self.df['cleaned_title'] + ' ' + self.df['cleaned_content']

        # Tokenize and lemmatize
        print("Tokenizing and lemmatizing...")
        self.df['processed_text'] = self.df['combined_text'].apply(self.tokenize_and_lemmatize)

        # Extract sentiment features
        print("Extracting sentiment features...")
        sentiment_features = self.df['combined_text'].apply(self.extract_sentiment_features)
        sentiment_df = pd.DataFrame(list(sentiment_features))

        # Combine with main dataframe
        self.df = pd.concat([self.df, sentiment_df], axis=1)

        print(f"Preprocessing complete. Shape: {self.df.shape}")
        return self.df

    def create_tfidf_features(self, max_features=1000):
        """Create TF-IDF features from processed text"""
        print(f"Creating TF-IDF features (max_features={max_features})...")

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )

        # Fit and transform
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['processed_text'])

        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(max_features)]
        )

        # Combine with main dataframe
        self.df = pd.concat([self.df, tfidf_df], axis=1)

        print(f"TF-IDF features created. New shape: {self.df.shape}")
        return self.df

    def aggregate_daily_sentiment(self):
        """Aggregate sentiment features by date for stock price integration"""
        print("Aggregating daily sentiment features...")

        # Group by date and calculate daily aggregates
        daily_sentiment = self.df.groupby('date').agg({
            'textblob_polarity': ['mean', 'std', 'count'],
            'custom_polarity': ['mean', 'std'],
            'positive_word_count': 'sum',
            'negative_word_count': 'sum',
            'text_length': 'mean',
            'word_count': 'sum'
        }).round(4)

        # Flatten column names
        daily_sentiment.columns = ['_'.join(col).strip() for col in daily_sentiment.columns.values]
        daily_sentiment = daily_sentiment.reset_index()

        # Rename columns for clarity
        daily_sentiment = daily_sentiment.rename(columns={
            'textblob_polarity_mean': 'avg_textblob_polarity',
            'textblob_polarity_std': 'std_textblob_polarity',
            'textblob_polarity_count': 'article_count',
            'custom_polarity_mean': 'avg_custom_polarity',
            'custom_polarity_std': 'std_custom_polarity',
            'positive_word_count_sum': 'total_positive_words',
            'negative_word_count_sum': 'total_negative_words',
            'text_length_mean': 'avg_article_length',
            'word_count_sum': 'total_words'
        })

        # Calculate additional derived features
        daily_sentiment['sentiment_intensity'] = daily_sentiment['avg_textblob_polarity'].abs()
        daily_sentiment['sentiment_volatility'] = daily_sentiment['std_textblob_polarity']
        daily_sentiment['pos_neg_ratio'] = (daily_sentiment['total_positive_words'] /
                                          daily_sentiment['total_negative_words'].replace(0, 1))

        self.daily_sentiment = daily_sentiment
        print(f"Daily sentiment aggregation complete. Shape: {daily_sentiment.shape}")

        return daily_sentiment

    def save_processed_data(self):
        """Save processed datasets"""
        print("Saving processed data...")

        # Save full processed dataset
        full_output = 'd:\\project_option_C\\project_option_c\\task7\\data_preprocessing\\processed_news_data.csv'
        self.df.to_csv(full_output, index=False)
        print(f"Saved full processed data: {full_output}")

        # Save daily sentiment features
        daily_output = 'd:\\project_option_C\\project_option_c\\task7\\data_preprocessing\\daily_sentiment_features.csv'
        self.daily_sentiment.to_csv(daily_output, index=False)
        print(f"Saved daily sentiment features: {daily_output}")

    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("=== Task 7 Data Preprocessing Pipeline ===\n")

        # Load data
        if not self.load_data():
            return False

        # Preprocess text
        self.preprocess_text_data()

        # Create TF-IDF features
        self.create_tfidf_features(max_features=500)

        # Aggregate daily sentiment
        self.aggregate_daily_sentiment()

        # Save results
        self.save_processed_data()

        print("\n=== Preprocessing Pipeline Complete ===")
        print(f"Processed {len(self.df)} articles into {len(self.daily_sentiment)} daily sentiment features")

        return True

def main():
    """Main function to run the preprocessing pipeline"""
    preprocessor = Task7DataPreprocessor()
    success = preprocessor.run_full_pipeline()

    if success:
        print("\nNext steps:")
        print("1. Review the processed data in data_preprocessing folder")
        print("2. Integrate with stock price data from Task 6")
        print("3. Train sentiment-enhanced ensemble models")
    else:
        print("Preprocessing failed. Please check data paths and dependencies.")

if __name__ == "__main__":
    main()