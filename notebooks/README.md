# Customer Experience Analytics for Fintech Apps

This project collects, cleans, and analyzes customer reviews for Ethiopian banking apps from Google Play.

## Project Structure

```
customer-experience-analytics-for-fintech-apps-week2/
│
├── scripts/
│   ├── data_collection.py   # Scrapes reviews from Google Play
│   ├── data_cleaner.py      # Cleans and summarizes review data
│   └── ...
├── notebooks/
│   └── data collection and preprocessing.ipynb
├── Ethiopian_bank_reviews.csv           # Raw reviews
├── cleaned_reviews.csv                  # Cleaned data
└── README.md
```

## Setup

1. **Clone the repository**
2. **Install dependencies**
   ```
   pip install pandas tqdm google-play-scraper
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
This creates `cleaned_reviews.csv` and prints a summary.

### 3. Analyze Data

Open and run the notebook:
```
notebooks/data collection and preprocessing.ipynb
```

## Customization

- Edit `BANK_APPS` in `data_collection.py` to add/remove apps.
- Change `REVIEWS_PER_APP` to adjust review count per app.

