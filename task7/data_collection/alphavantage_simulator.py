"""
Alpha Vantage News API Client for META Stock News
Fetches financial news data about META stock using Alpha Vantage News API.

This script demonstrates how to use Alpha Vantage News API to collect news articles
about META stock for sentiment analysis.

Output: alphavantage_meta_news.csv with collected news data
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os

class AlphaVantageNewsClient:
    def __init__(self, api_key):
        """
        Initialize Alpha Vantage News API client with API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch_news(self, tickers=None, topics=None, time_from=None, time_to=None, limit=50):
        """
        Fetch news articles using Alpha Vantage News API
        """
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.api_key,
            'limit': min(limit, 1000)
        }

        if tickers:
            params['tickers'] = ','.join(tickers)
        if topics:
            params['topics'] = ','.join(topics)
        if time_from:
            params['time_from'] = time_from
        if time_to:
            params['time_to'] = time_to

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return {"feed": []}

    def collect_meta_news(self, start_date="2023-01-01", end_date="2024-07-02"):
        """
        Collect news articles about META stock
        """
        all_news_items = []

        # Define parameters for META news
        tickers = ["META", "FB"]
        topics = ["financial_markets", "earnings", "technology"]

        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date < end_datetime:
            chunk_end = min(current_date + timedelta(days=30), end_datetime)

            time_from = current_date.strftime('%Y%m%dT0000')
            time_to = chunk_end.strftime('%Y%m%dT2359')

            print(f"Fetching news for META from {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

            response = self.fetch_news(
                tickers=tickers,
                topics=topics,
                time_from=time_from,
                time_to=time_to,
                limit=50
            )

            if "feed" in response:
                all_news_items.extend(response["feed"])

            # Rate limiting
            time.sleep(1)

            current_date = chunk_end

        return all_news_items

def main():
    """Main function to collect and save news data"""
    # Note: Replace with actual API key for real usage
    API_KEY = os.getenv('ALPHAVANTAGE_API_KEY', 'YOUR_ALPHAVANTAGE_API_KEY_HERE')

    print("Starting Alpha Vantage News collection for META stock news...")

    client = AlphaVantageNewsClient(API_KEY)

    # Collect news data
    news_items = client.collect_meta_news()

    # Convert to DataFrame
    df_data = []
    for item in news_items:
        # Extract META-specific sentiment if available
        meta_sentiment = None
        relevance_score = None
        if "ticker_sentiment" in item:
            for ticker_data in item["ticker_sentiment"]:
                if ticker_data.get("ticker") == "META":
                    meta_sentiment = float(ticker_data.get("ticker_sentiment_score", 0))
                    relevance_score = float(ticker_data.get("relevance_score", 0.5))
                    break

        # Use overall sentiment if META-specific not found
        if meta_sentiment is None:
            meta_sentiment = float(item.get("overall_sentiment_score", 0))

        df_data.append({
            'date': item.get('time_published', '')[:8] if item.get('time_published') else '',  # YYYYMMDD format
            'title': item.get('title', ''),
            'content': item.get('summary', ''),
            'source': item.get('source', ''),
            'url': item.get('url', ''),
            'author': ', '.join(item.get('authors', [])),
            'sentiment_polarity': meta_sentiment,
            'relevance_score': relevance_score or 0.5,
            'sentiment_label': item.get('overall_sentiment_label', '')
        })

    df = pd.DataFrame(df_data)

    # Convert date format and clean data
    if len(df) > 0:
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
        df = df.dropna(subset=['date'])

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['title', 'date']).sort_values('date').reset_index(drop=True)

    # Save to CSV
    output_path = 'd:\\project_option_C\\project_option_c\\task7\\data_collection\\alphavantage_meta_news.csv'
    df.to_csv(output_path, index=False)

    print(f"Collected {len(df)} articles from Alpha Vantage News API")
    print(f"Saved to: {output_path}")

    if len(df) > 0:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("Sample articles:")
        print(df.head(3)[['date', 'title', 'source', 'sentiment_polarity', 'sentiment_label']])

if __name__ == "__main__":
    main()