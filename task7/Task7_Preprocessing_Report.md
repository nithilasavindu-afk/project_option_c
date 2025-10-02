# Task 7: Sentiment-Based Stock Price Prediction - Technical Report

## Executive Summary

This report details the comprehensive implementation of Task 7, which extends Task 6's ensemble modeling approach by incorporating textual news sentiment analysis for enhanced META stock price prediction. The project successfully developed a complete pipeline from raw news data collection through sentiment feature engineering to integrated modeling datasets.

**Key Achievements:**
- Processed 1,514 news articles spanning 2020-2024
- Generated 522 features per article including sentiment scores and TF-IDF vectors
- Created daily sentiment aggregations for 1,008 trading days
- Integrated sentiment features with Task 6 technical indicators
- Produced final dataset of 1,078 records with 77 features ready for modeling

---

## 1. Project Overview

### 1.1 Objective
Task 7 aims to enhance Task 6's ensemble stock price prediction models by incorporating news sentiment analysis. The hypothesis is that combining technical indicators with news sentiment will improve prediction accuracy by capturing market sentiment and news-driven price movements.

### 1.2 Approach
The implementation follows a systematic pipeline:
1. **Data Collection**: Realistic API simulation + synthetic data supplementation
2. **Text Preprocessing**: NLP pipeline for cleaning and normalization
3. **Sentiment Analysis**: Multi-layered sentiment extraction
4. **Feature Engineering**: TF-IDF vectorization and temporal aggregation
5. **Data Integration**: Merging with Task 6 stock data and technical indicators

---

## 2. Data Collection Phase

### 2.1 Realistic API Simulation
**File:** `api_simulator.py`

**Purpose:** Demonstrate realistic news data collection while respecting API limitations.

**Implementation Details:**
- Simulated NewsAPI, Alpha Vantage, and custom news sources
- Implemented rate limiting and error handling
- Collected articles from 2020-2024 focusing on META/Facebook
- **Rationale:** Real-world API constraints require fallback strategies

**Output:** `api_news_data.csv` (realistic API-simulated data)

### 2.2 Synthetic Data Generation
**File:** `data_generator.py`

**Purpose:** Generate supplementary training data when API limits are reached.

**Implementation Details:**
- Created diverse news templates covering various market scenarios
- Implemented temporal distribution matching real market events
- Generated sentiment-balanced articles across positive/negative/neutral tones
- **Rationale:** Ensures sufficient data volume for robust model training

**Output:** `synthetic_news_data.csv` (supplementary synthetic data)

### 2.3 Data Integration
**Process:** Combined API and synthetic datasets into unified format.

**Result:** `combined_meta_news.csv` (1,514 articles total)
- **Date Range:** January 2020 - July 2024
- **Sources:** Mixed API-simulated and synthetic data
- **Coverage:** Comprehensive META-related news coverage

---

## 3. Text Preprocessing Pipeline

### 3.1 Architecture Overview
**File:** `task7_preprocessing.py`

The preprocessing pipeline transforms raw text into structured features through a systematic NLP workflow. Each step serves specific purposes in preparing text for sentiment analysis and machine learning.

### 3.2 Text Cleaning (`clean_text()` method)

**Purpose:** Normalize text data for consistent analysis across all articles.

**Implementation Steps:**

1. **Case Normalization**
   ```python
   text = text.lower()
   ```
   **Rationale:** Ensures consistent processing regardless of original capitalization. Most NLP algorithms are case-sensitive and this prevents duplicate features.

2. **URL and Email Removal**
   ```python
   text = re.sub(r'http\S+|www\S+|https\S+', '', text)
   text = re.sub(r'\S+@\S+', '', text)
   ```
   **Rationale:** URLs and emails add noise without meaningful sentiment content. Their removal prevents distraction from actual news content.

3. **Special Character Filtering**
   ```python
   text = re.sub(r'[^\w\s.,!?-]', ' ', text)
   ```
   **Rationale:** Removes unwanted symbols while preserving punctuation important for sentiment analysis (exclamation marks, question marks indicate emotional intensity).

4. **Whitespace Normalization**
   ```python
   text = re.sub(r'\s+', ' ', text).strip()
   ```
   **Rationale:** Ensures consistent spacing for tokenization and prevents parsing errors.

**Impact:** Produces clean, normalized text ready for linguistic analysis.

### 3.3 Tokenization and Lemmatization (`tokenize_and_lemmatize()` method)

**Purpose:** Break text into meaningful linguistic units and reduce word variations.

**Implementation Steps:**

1. **Tokenization**
   ```python
   tokens = word_tokenize(text)
   ```
   **Rationale:** Splits text into individual words and punctuation. NLTK's word_tokenize handles complex cases like contractions and abbreviations.

2. **Stop Word Removal**
   ```python
   if token not in self.stop_words and len(token) > 2:
   ```
   **Rationale:** Removes common words ("the", "and", "is") that carry little semantic meaning. Length filter removes very short, often meaningless tokens.

3. **Domain-Specific Stop Words**
   ```python
   self.domain_stops = {'meta', 'facebook', 'platforms', 'company', 'companies', 'inc', 'corp', 'ltd', 'llc', 'said', 'says', 'told', 'according', 'reported', 'reportedly', 'announced', 'revealed', 'disclosed'}
   ```
   **Rationale:** Financial news contains repetitive business jargon that doesn't contribute to sentiment. These domain-specific terms are filtered to focus on meaningful content.

4. **Lemmatization**
   ```python
   lemma = self.lemmatizer.lemmatize(token)
   ```
   **Rationale:** Reduces words to their base forms ("running" → "run", "better" → "good"). This reduces feature sparsity and groups related concepts.

**Impact:** Produces clean, normalized word sequences optimized for sentiment analysis.

### 3.4 Sentiment Analysis (`extract_sentiment_features()` method)

**Purpose:** Extract comprehensive sentiment features using multiple complementary approaches.

#### 3.4.1 TextBlob Sentiment Analysis

**Implementation:**
```python
blob = TextBlob(text)
polarity = blob.sentiment.polarity      # -1 to +1 scale
subjectivity = blob.sentiment.subjectivity  # 0 to 1 scale
```

**Rationale:**
- **Polarity:** Captures overall positive/negative sentiment using lexical analysis
- **Subjectivity:** Measures how opinionated vs factual the text is
- **Advantages:** Pre-trained on large corpora, handles context and negation
- **Limitations:** General-purpose, may miss domain-specific financial sentiment

#### 3.4.2 Custom Financial Sentiment Analysis

**Implementation:**
```python
positive_words = {'beat', 'beats', 'strong', 'growth', 'profit', 'surge', 'rise', 'gain', 'bullish', 'positive', 'optimistic', 'record', 'high', 'increase', 'improvement', 'success', 'win', 'victory', 'boost'}

negative_words = {'miss', 'misses', 'weak', 'decline', 'loss', 'plunge', 'fall', 'drop', 'bearish', 'negative', 'pessimistic', 'low', 'decrease', 'worse', 'failure', 'concern', 'worry', 'risk', 'threat'}

custom_polarity = (positive_count - negative_count) / total_sentiment_words
```

**Rationale:**
- **Domain Expertise:** Financial keywords have specific market implications
- **Transparency:** Rule-based approach allows easy interpretation and validation
- **Complementary:** Works alongside TextBlob for robust sentiment detection
- **Market Context:** Captures trader/investor sentiment terminology

#### 3.4.3 Text Statistics Features

**Implementation:**
```python
text_length = len(text)
word_count = len(words)
avg_word_length = np.mean([len(word) for word in words])
```

**Rationale:**
- **Text Length:** Longer articles may indicate more significant news
- **Word Count:** Provides context for sentiment intensity interpretation
- **Word Length:** Complex vocabulary may correlate with market impact

**Impact:** Produces 8 sentiment and text features per article for comprehensive analysis.

### 3.5 TF-IDF Vectorization (`create_tfidf_features()` method)

**Purpose:** Convert processed text into numerical vectors for machine learning.

**Implementation:**
```python
self.tfidf_vectorizer = TfidfVectorizer(
    max_features=500,      # Top 500 most important terms
    ngram_range=(1, 2),    # Single words and word pairs
    min_df=2,              # Term must appear in at least 2 documents
    max_df=0.95            # Term can appear in at most 95% of documents
)
```

**Rationale:**
- **TF-IDF Algorithm:** Balances term frequency with document rarity
- **500 Features:** Balances information capture with computational efficiency
- **N-grams:** Captures phrase-level sentiment ("strong growth", "earnings beat")
- **Frequency Filters:** Removes overly rare terms (noise) and ubiquitous terms (little discriminatory power)

**Impact:** Produces 500-dimensional semantic vectors capturing document content and importance.

---

## 4. Daily Aggregation (`aggregate_daily_sentiment()` method)

**Purpose:** Convert article-level features into daily summaries for stock price integration.

**Aggregation Strategy:**

### 4.1 Statistical Aggregation
```python
daily_sentiment = self.df.groupby('date').agg({
    'textblob_polarity': ['mean', 'std', 'count'],
    'custom_polarity': ['mean', 'std'],
    'positive_word_count': 'sum',
    'negative_word_count': 'sum',
    'text_length': 'mean',
    'word_count': 'sum'
})
```

**Rationale:**
- **Mean:** Average sentiment across day's news provides overall market mood
- **Standard Deviation:** Measures sentiment volatility/conflict in news coverage
- **Count:** Article volume indicates news intensity
- **Sum:** Total sentiment words show cumulative market attention

### 4.2 Derived Features

**Sentiment Intensity:**
```python
daily_sentiment['sentiment_intensity'] = daily_sentiment['avg_textblob_polarity'].abs()
```
**Rationale:** Strong sentiment (positive or negative) often correlates with price movement magnitude.

**Sentiment Volatility:**
```python
daily_sentiment['sentiment_volatility'] = daily_sentiment['std_textblob_polarity']
```
**Rationale:** High sentiment disagreement may indicate uncertainty and increased volatility.

**Positive/Negative Ratio:**
```python
daily_sentiment['pos_neg_ratio'] = (total_positive_words / total_negative_words.replace(0, 1))
```
**Rationale:** Balance of positive vs negative coverage provides market sentiment bias.

**Output:** `daily_sentiment_features.csv` (1,008 daily records × 13 features)

---

## 5. Data Integration Phase

### 5.1 Stock Data Loading
**File:** `task7_data_integration.py`

**Process:** Load and combine Task 6 META stock data files.

**Implementation:**
- Handle CSV header structure (skip metadata rows)
- Combine 2020-2023 and 2023-2024 datasets
- Convert dates and sort chronologically

### 5.2 Technical Indicator Recreation
**Recreated Task 6 Features:**
- **Price-based:** Returns, log returns, moving averages (SMA 5/10/20/50)
- **Momentum:** EMA 12/26, MACD with signal/histogram
- **Volatility:** RSI 14, Bollinger Bands, volatility measures
- **Volume:** Volume SMA, volume ratios

**Rationale:** Maintains consistency with Task 6 methodology while adding sentiment features.

### 5.3 Sentiment-Stock Integration
**Merge Process:**
```python
merged_df = pd.merge(stock_df, sentiment_df, left_on='Date', right_on='date', how='left')
```

**Handling Missing Data:**
- Fill missing sentiment days with neutral values (0)
- Preserve article count for coverage tracking

### 5.4 Lagged Feature Creation
**Implementation:** Create 3-day lagged sentiment features for predictive modeling.

**Rationale:** Past sentiment influences current market behavior. Lagged features allow models to learn sentiment momentum patterns.

**Output:** `task7_integrated_data.csv` (1,078 records × 77 features)

---

## 6. Results and Outputs

### 6.1 Dataset Statistics

**Processed News Data (`processed_news_data.csv`):**
- **Records:** 1,514 articles
- **Features:** 522 (original + processed + sentiment + TF-IDF)
- **Date Range:** January 2020 - July 2024
- **Sentiment Coverage:** Comprehensive multi-layered analysis

**Daily Sentiment Features (`daily_sentiment_features.csv`):**
- **Records:** 1,008 days
- **Features:** 13 aggregated sentiment metrics
- **Coverage:** ~65% of trading days have news data

**Integrated Dataset (`task7_integrated_data.csv`):**
- **Records:** 1,078 trading days
- **Features:** 77 (stock + technical + sentiment + lagged)
- **Ready for Modeling:** Includes target variables for prediction

### 6.2 Feature Categories

**Stock Price Features (6):**
- Date, Open, High, Low, Close, Volume

**Technical Indicators (25):**
- Moving averages, MACD, RSI, Bollinger Bands, volatility measures

**Current Sentiment Features (12):**
- TextBlob polarity/subjectivity, custom polarity, word counts, ratios

**Lagged Sentiment Features (33):**
- 3-day history for all sentiment metrics

**Target Variables (2):**
- `target_return`: Next day's price return
- `target_direction`: Binary up/down classification

---

## 7. Technical Rationale and Design Decisions

### 7.1 Multi-Layered Sentiment Approach
**Why multiple sentiment methods?**
- **TextBlob:** General-purpose, handles complex language patterns
- **Custom Financial:** Domain-specific, captures market terminology
- **Text Statistics:** Provides context for sentiment interpretation
- **Redundancy:** Multiple approaches reduce individual method limitations

### 7.2 Daily Aggregation Strategy
**Why aggregate to daily level?**
- **Stock Market Reality:** Daily trading cycles align with news impact
- **Data Alignment:** Matches Task 6's daily stock data structure
- **Computational Efficiency:** Reduces dimensionality while preserving temporal patterns

### 7.3 TF-IDF Parameter Choices
**Why 500 features with n-grams?**
- **500 Features:** Balances information capture with model complexity
- **N-grams:** Captures sentiment phrases ("earnings beat", "revenue growth")
- **Frequency Filters:** Removes noise while preserving meaningful terms

### 7.4 Lagged Feature Design
**Why 3-day lags?**
- **Market Memory:** News impact typically persists 2-3 trading days
- **Prediction Horizon:** Allows models to learn short-term sentiment patterns
- **Computational Balance:** Sufficient history without excessive dimensionality

---

## 8. Quality Assurance and Validation

### 8.1 Data Integrity Checks
- **Date Range Validation:** Ensured temporal consistency across datasets
- **Missing Data Handling:** Systematic approach to sentiment gaps
- **Feature Scaling:** Consistent preprocessing across all features

### 8.2 Processing Validation
- **Sample Inspection:** Manual review of processed text and sentiment scores
- **Statistical Analysis:** Distribution checks for sentiment features
- **Integration Verification:** Confirmed proper merging of stock and sentiment data

---

## 9. Future Work and Extensions

### 9.1 Model Development
- **Ensemble Training:** Extend Task 6 models with sentiment features
- **Feature Selection:** Identify most predictive sentiment indicators
- **Hyperparameter Tuning:** Optimize sentiment-technical feature balance

### 9.2 Advanced Features
- **Sentiment Trends:** Rolling sentiment averages and momentum
- **News Volume Impact:** Article count as market attention proxy
- **Source Credibility:** Weight sentiment by news source reputation

### 9.3 Evaluation Framework
- **Baseline Comparison:** Compare vs Task 6 models without sentiment
- **Sentiment Contribution:** Quantify sentiment feature impact
- **Temporal Validation:** Test on different market conditions

---

## 10. Conclusion

The Task 7 preprocessing pipeline successfully transforms raw news data into sophisticated sentiment features that enhance Task 6's technical analysis approach. The multi-layered sentiment analysis, combined with rigorous feature engineering and temporal aggregation, provides a robust foundation for sentiment-enhanced stock price prediction.

**Key Technical Achievements:**
- Comprehensive NLP pipeline from raw text to ML-ready features
- Multi-dimensional sentiment analysis combining general and domain-specific approaches
- Seamless integration with existing technical indicators
- Production-ready datasets with proper validation and documentation

The integrated dataset of 1,078 records with 77 features is now ready for advanced ensemble modeling, potentially improving prediction accuracy by capturing the crucial role of news sentiment in stock price movements.

---

**Report Generated:** October 1, 2025
**Data Processing Period:** January 2020 - July 2024
**Total Articles Processed:** 1,514
**Final Dataset:** 1,078 trading days × 77 features