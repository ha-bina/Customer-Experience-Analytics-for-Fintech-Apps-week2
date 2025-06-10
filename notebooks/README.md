# Customer Experience Analytics for Fintech Apps

This project collects, cleans, analyzes, and stores customer reviews for Ethiopian banking apps from Google Play.

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
│   ├── data collection and preprocessing.ipynb
│   └── database_creation.py    # Creates SQLite DB and inserts cleaned data
├── Ethiopian_bank_reviews.csv              # Raw collected reviews
├── Ethiopian_banks_review_cleaned.csv      # Cleaned review data
├── bank_reviews.db                         # SQLite database
└── README.md
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install pandas tqdm google-play-scraper scikit-learn nltk textblob spacy wordcloud torch transformers
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

### 3. Analyze Data

Open and run the notebook:
```
notebooks/data collection and preprocessing.ipynb
```

### 4. Create Database and Insert Data

Run:
```
python notebooks/database_creation.py
```
This creates `bank_reviews.db` and inserts the cleaned reviews.

## Database Schema

- **Banks Table:** Stores bank names.
- **Reviews Table:** Stores reviews, linked to banks.

## Customization

- Edit `BANK_APPS` in `data_collection.py` to add/remove apps.
- Change `REVIEWS_PER_APP` to adjust review count per app.


