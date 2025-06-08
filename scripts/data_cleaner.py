import pandas as pd
import time
from tqdm import tqdm
import os
# Ethiopian bank apps to scrape
BANK_APPS = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking", 
    "Dashen Bank": "com.dashen.dashensuperapp"
}
#Configurations
REVIEWS_PER_APP = 400
LANGUAGE = 'en'
SORT = Sort.NEWEST
DELAY = 2  # Avoid rate-limiting

def fetch_reviews(app_id, app_name):
    """Scrape reviews with error handling and progress tracking"""
    reviews_data = []
    continuation_token = None
    
    with tqdm(total=REVIEWS_PER_APP, desc=f"Scraping {app_name}") as pbar:
        while len(reviews_data) < REVIEWS_PER_APP:
            try:
                batch, token = reviews(
                    app_id,
                    lang=LANGUAGE,
                    sort=SORT,
                    count=100,
                    continuation_token=continuation_token
                )
                if not batch:
                    break
                reviews_data.extend(batch)
                pbar.update(len(batch))
                time.sleep(DELAY)
                continuation_token = token
            except Exception as e:
                print(f"Error: {e}")
                break
    return reviews_data[:REVIEWS_PER_APP]

def clean_data(df):
    """Data cleaning pipeline"""
    # Handle duplicates
    df = df.drop_duplicates(subset=['review'], keep='first')
    
    # Handle missing data
    df['review'] = df['review'].fillna('(No text)')
    df['rating'] = df['rating'].fillna(0).astype(int)
    
    # Normalize dates
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    
    return df

def main():
    all_reviews = []
    
    for bank_name, app_id in BANK_APPS.items():
        print(f"\nProcessing {bank_name}...")
        
        # Scrape reviews
        raw_reviews = fetch_reviews(app_id, bank_name)
        
        # Structure data
        for review in raw_reviews:
            all_reviews.append({
                'review': review.get('content', ''),
                'rating': review.get('score', 0),
                'date': review.get('at', pd.NaT),
                'bank': bank_name,
                'source': 'Google Play'
            })
    
    # Create DataFrame and clean
    df = pd.DataFrame(all_reviews)
    df = clean_data(df)
    
    # Save to CSV
    output_file = "bank_reviews_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"\nSuccess! Saved {len(df)} reviews to {output_file}")
    
    # Show summary
    print("\nData Overview:")
    print(f"Total Reviews: {len(df)}")
    print(f"Reviews per Bank:\n{df['bank'].value_counts()}")
    print(f"Rating Distribution:\n{df['rating'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()