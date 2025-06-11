# Customer Experience Analytics for Fintech Apps

This project collects, cleans, analyzes, and visualizes customer reviews for Ethiopian banking apps from Google Play. It includes sentiment analysis, thematic analysis, and database storage.

## Project Structure

```
customer-experience-analytics-for-fintech-apps-week2/
│
├── scripts/
│   ├── data_collection.py      # Scrapes reviews from Google Play
│   ├── data_cleaner.py         # Cleans and standardizes review data
│   ├── sentiment_analysis.py   # Sentiment analysis and keyword extraction
│   ├── thematic_analysis.py    # Thematic clustering and analysis
│   └── ...
├── notebooks/
│   ├── Customer_experiance_analysis.ipynb  # Main analysis notebook
│   ├── data collection and preprocessing.ipynb
│   └── database_creation.py    # Creates SQLite DB and inserts cleaned data
├── Ethiopian_bank_reviews.csv              # Raw collected reviews
├── Ethiopian_banks_review_cleaned.csv      # Cleaned review data
├── sentiment_analysis_aggregated.csv       # Sentiment aggregation results
├── bank_reviews.db                         # SQLite database
└── README.md
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install pandas tqdm google-play-scraper scikit-learn nltk textblob spacy wordcloud torch transformers matplotlib
   python -m spacy download en_core_web_sm
   ```

## Usage

### 1. Collect Reviews

Run:
```
python scripts/data_collection.py
```
This creates `Ethiopian_bank_reviews.csv`.

### 2. Clean Data

Run:
```
python scripts/data_cleaner.py
```
This creates `Ethiopian_banks_review_cleaned.csv`.

### 3. Sentiment Analysis

Run:
```
python scripts/sentiment_analysis.py
```
Or use the provided notebook for step-by-step analysis and visualization.

### 4. Thematic Analysis

Run:
```
python scripts/thematic_analysis.py
```

### 5. Database Creation

Run:
```
python notebooks/database_creation.py
```
This creates `bank_reviews.db` and inserts the cleaned reviews.

### 6. Visualization

Open and run the notebook:
```
notebooks/Customer_experiance_analysis.ipynb
```
This notebook includes:
- Sentiment distribution plots per app and per rating group
- Word clouds for positive/negative reviews
- Grouped bar charts for selected banks

## Database Schema

- **Banks Table:** Stores bank names.
- **Reviews Table:** Stores reviews, linked to banks.

## Customization

- Edit `BANK_APPS` in `data_collection.py` to add/remove apps.
- Change `REVIEWS_PER_APP` to adjust review count per app.
- Adjust plotting code in the notebook for different visualizations.

