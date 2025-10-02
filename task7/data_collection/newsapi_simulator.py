"""
NewsAPI Client for META Stock News
Fetches financial news data about META stock using NewsAPI.

This script demonstrates how to use NewsAPI to collect news articles
about META stock for sentiment analysis.

Output: newsapi_meta_news.csv with collected news data
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time

class NewsAPIClient:
    def __init__(self, api_key):
        """
        Initialize NewsAPI client with API key
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query, from_date, to_date, max_results=20):
        """
        Fetch news articles using NewsAPI
        """
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'publishedAt',
            'apiKey': self.api_key,
            'pageSize': min(max_results, 100),
            'language': 'en'
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news: {e}")
            return {"status": "error", "articles": []}

    def collect_meta_news(self, start_date="2023-01-01", end_date="2024-07-02"):
        """
        Collect news articles about META stock
        """
        all_articles = []

        # Define search queries for META
        queries = [
            "META stock",
            "Facebook earnings",
            "Meta Platforms"
        ]

        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_datetime = datetime.strptime(end_date, '%Y-%m-%d')

        while current_date < end_datetime:
            chunk_end = min(current_date + timedelta(days=7), end_datetime)

            for query in queries:
                print(f"Fetching news for '{query}' from {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")

                response = self.fetch_news(
                    query=query,
                    from_date=current_date.strftime('%Y-%m-%d'),
                    to_date=chunk_end.strftime('%Y-%m-%d'),
                    max_results=10
                )

                if response.get("status") == "ok" and "articles" in response:
                    all_articles.extend(response["articles"])

                # Rate limiting
                time.sleep(1)

            current_date = chunk_end

        return all_articles

def main():
    """Main function to collect and save news data"""
    # Note: Replace with actual API key for real usage
    API_KEY = "YOUR_NEWSAPI_KEY_HERE"

    print("Starting NewsAPI collection for META stock news...")

    client = NewsAPIClient(API_KEY)

    # Collect news data
    articles = client.collect_meta_news()

    # Convert to DataFrame
    df_data = []
    for article in articles:
        df_data.append({
            'date': article.get('publishedAt', '')[:10] if article.get('publishedAt') else '',
            'title': article.get('title', ''),
            'content': article.get('content') or article.get('description', ''),
            'source': article.get('source', {}).get('name', ''),
            'url': article.get('url', ''),
            'author': article.get('author', '')
        })

    df = pd.DataFrame(df_data)

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=['title', 'date']).sort_values('date').reset_index(drop=True)

    # Save to CSV
    output_path = 'd:\\project_option_C\\project_option_c\\task7\\data_collection\\newsapi_meta_news.csv'
    df.to_csv(output_path, index=False)

    print(f"Collected {len(df)} articles from NewsAPI")
    print(f"Saved to: {output_path}")

    if len(df) > 0:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
        print("Sample articles:")
        print(df.head(3)[['date', 'title', 'source']])

if __name__ == "__main__":
    main()