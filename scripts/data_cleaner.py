import pandas as pd
from datetime import datetime

def clean_reviews(input_file, output_file):
    """
    Processes review data to:
    - Remove duplicates
    - Handle missing values
    - Normalize dates
    - Standardize output columns
    """
    try:
        # Load data with flexible column matching
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} records from {input_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Standardize column names
    col_map = {
        'app name': ['app name', 'app', 'application', 'bank'],
        'review text': ['review text', 'review', 'text', 'comment'],
        'review date': ['review date', 'date', 'timestamp'],
        'rating': ['rating', 'score', 'stars']
    }

    for standard_name, variants in col_map.items():
        for variant in variants:
            if variant.lower() in [col.lower() for col in df.columns]:
                if standard_name not in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                break
        if standard_name not in df.columns:
            df[standard_name] = pd.NA

    # Data cleaning pipeline
    def clean_data(df):
        # Remove exact duplicate reviews
        df = df.drop_duplicates(subset=['review text'], keep='first')
        
        # Handle missing values
        df['review text'] = df['review text'].fillna('(No review text)')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
        
        # Normalize dates to YYYY-MM-DD
        df['review date'] = pd.to_datetime(
            df['review date'],
            errors='coerce',
            format='mixed'
        ).dt.strftime('%Y-%m-%d')
        
        # Clean app names
        df['app name'] = df['app name'].str.strip().str.title()
        
        return df

    cleaned_df = clean_data(df)

    # Filter to only keep records with valid dates and app names
    final_df = cleaned_df[
        (cleaned_df['review date'].notna()) &
        (cleaned_df['app name'].notna())
    ].copy()

    # Save to CSV with specified columns
    try:
        final_df.to_csv(
            output_file,
            index=False,
            columns=['app_name', 'review_text', 'review_date', 'rating']
        )
        print(f"Saved {len(final_df)} cleaned records to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return

    # Generate summary report
    def generate_summary(df):
        return {
            'total_reviews': len(df),
            'apps': sorted(df['app name'].unique()),
            'date_range': f"{df['review date'].min()} to {df['review date'].max()}",
            'rating_stats': df['rating'].value_counts().sort_index().to_dict(),
            'missing_data': {
                'reviews': sum(df['review text'] == '(No review text)'),
                'ratings': sum(df['rating'] == 0)
            }
        }

    report = generate_summary(final_df)
    
    print("\n=== CLEANING REPORT ===")
    print(f"Final Review Count: {report['total_reviews']}")
    print(f"Apps Included: {', '.join(report['apps'])}")
    print(f"Date Range: {report['date_range']}")
    print("\nRating Distribution:")
    for rating, count in report['rating_stats'].items():
        print(f"- {rating}-star: {count}")
    print(f"\nMissing Data:")
    print(f"- Empty reviews: {report['missing_data']['reviews']}")
    print(f"- Zero ratings: {report['missing_data']['ratings']}")

def main():
    INPUT_FILE = "raw_reviews.csv"  # Change to your input file
    OUTPUT_FILE = "cleaned_reviews.csv"  # Output filename
    
    print("Starting data cleaning process...")
    clean_reviews(INPUT_FILE, OUTPUT_FILE)
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()