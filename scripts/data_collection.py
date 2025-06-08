#%pip install google-play-scraper
#%pip install tqdm
#%pip install pandas
from google_play_scraper import app, reviews, Sort
import pandas as pd
import time
from tqdm import tqdm
import os
# Ethiopian bank apps to scrape
BANK_APPS = {
    "Commercial Bank of Ethiopia": "https://play.google.com/store/apps/details?id=com.combanketh.mobilebanking&hl=en",
    "Bank of Abyssinia": "https://play.google.com/store/apps/details?id=com.boa.boaMobileBanking&hl=en", 
    "Dashen Bank": "https://play.google.com/store/apps/details?id=com.dashen.dashensuperapp&hl=en"
}

# Configuration
REVIEWS_PER_APP = 400  # Target number of reviews per app
LANGUAGE = 'en'        # English reviews
SORT = Sort.NEWEST     # Get most recent reviews first
DELAY = 2             # Seconds between requests to avoid blocking

def fetch_app_details(app_id):
    """Get basic app information including name"""
    try:
        app_data = app(app_id, lang=LANGUAGE)
        return {
            'app_name': app_data['title'],
            'average_rating': app_data['score'],
            'total_reviews': app_data['reviews']
        }
    except Exception as e:
        print(f"Error fetching app {app_id}: {str(e)}")
        return None

def scrape_app_reviews(app_id, target_count):
    """Collect reviews for a specific app"""
    collected_reviews = []
    token = None
    
    with tqdm(total=target_count, desc="Collecting reviews") as progress:
        while len(collected_reviews) < target_count:
            try:
                batch, token = reviews(
                    app_id,
                    lang=LANGUAGE,
                    sort=SORT,
                    count=100,
                    continuation_token=token
                )
                
                if not batch:
                    break
                
                collected_reviews.extend(batch)
                progress.update(len(batch))
                time.sleep(DELAY)
                
            except Exception as e:
                print(f"Error during scraping: {str(e)}")
                break
    
    return collected_reviews[:target_count]

def format_review_data(reviews_list, app_name):
    """Format the review data with required fields"""
    formatted = []
    for review in reviews_list:
        review_date = review.get('at')
        if review_date is not None:
            review_date = review_date.date()
        else:
            review_date = None
        formatted.append({
            'app_name': app_name,
            'review_text': review.get('content', ''),
            'date': review_date,  # Extract just the date portion
            'rating': review.get('score', None)
        })
    return formatted

def main():
    all_reviews = []
    
    for bank_name, app_id in BANK_APPS.items():
        print(f"\nProcessing: {bank_name}")
        
        # Get app metadata
        app_info = fetch_app_details(app_id)
        if not app_info:
            continue
            
        print(f"Found app: {app_info['app_name']}")
        print(f"Average rating: {app_info['average_rating']}/5")
        print(f"Total reviews available: {app_info['total_reviews']}")
        
        # Scrape reviews
        app_reviews = scrape_app_reviews(app_id, REVIEWS_PER_APP)
        processed_reviews = format_review_data(app_reviews, app_info['app_name'])
        all_reviews.extend(processed_reviews)
        
        print(f"Successfully collected {len(processed_reviews)} reviews")
    
    # Create and save DataFrame
    df = pd.DataFrame(all_reviews)
    
    # Ensure proper date formatting, handle invalid dates gracefully
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # Save to CSV
    timestamp = pd.Timestamp.now().strftime("%Y%m%d")
    filename = f"ethiopian_bank_reviews_{timestamp}.csv"
    df.to_csv(filename, index=False)
    
    print("\n" + "="*50)
    print(f"Successfully saved {len(df)} total reviews to {filename}")
    print("="*50)
    
    # Show summary
    print("\nReview Summary:")
    print(f"- Total reviews: {len(df)}")
    print("- Reviews per app:")
    print(df['app_name'].value_counts())
    print("\n- Rating distribution:")
    print(df['rating'].value_counts().sort_index())
if __name__ == "__main__":
        main() 