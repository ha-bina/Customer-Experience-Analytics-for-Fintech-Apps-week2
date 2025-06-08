import pandas as pd
from datetime import datetime

def clean_review_data(input_file, output_file):
    """
    Processes bank review data to include all banks and all dates,
    with comprehensive cleaning and standardization.
    """
    # Load the data with error handling
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} raw records from {input_file}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Standardize column names (case insensitive)
    col_mapping = {
        'review': ['review', 'text', 'comment', 'content', 'feedback'],
        'rating': ['rating', 'score', 'stars', 'rating_score'],
        'date': ['date', 'review_date', 'time', 'timestamp'],
        'bank': ['bank', 'bank_name', 'app', 'application']
    }
    
    for standard_name, variants in col_mapping.items():
        for variant in variants:
            if variant.lower() in [col.lower() for col in df.columns]:
                if standard_name not in df.columns:
                    df.rename(columns={variant: standard_name}, inplace=True)
                break
        if standard_name not in df.columns:
            df[standard_name] = pd.NA

    # Ensure source column exists
    if 'source' not in df.columns:
        df['source'] = 'Google Play'

    # Data cleaning transformations
    def transform_data(df):
        # Handle duplicates - keep first occurrence
        df = df.drop_duplicates(subset=['review'], keep='first')
        
        # Clean review text
        df['review'] = df['review'].fillna('(No text)').str.strip()
        
        # Clean and standardize ratings (1-5)
        df['rating'] = (
            pd.to_numeric(df['rating'], errors='coerce')
            .clip(1, 5)
            .fillna(0)
            .astype(int)
        )
        
        # Normalize dates (handle multiple formats)
        df['date'] = pd.to_datetime(
            df['date'],
            errors='coerce',
            format='mixed'
        ).dt.strftime('%Y-%m-%d')
        
        # Standardize bank names
        bank_mapping = {
            'cbe': 'CBE',
            'com.cbe.': 'CBE',
            'bank of abyssinia': 'BOA',
            'boa': 'BOA',
            'dashen': 'Dashen'
        }
        df['bank'] = (
            df['bank'].str.lower().str.strip()
            .replace(bank_mapping)
            .str.upper()
        )
        
        return df

    # Apply cleaning
    cleaned_df = transform_data(df)
    
    # Filter to only include valid data
    valid_df = cleaned_df[
        (cleaned_df['bank'].notna()) &
        (cleaned_df['date'].notna())
    ].copy()
    
    # Save all valid records
    try:
        valid_df.to_csv(output_file, index=False, columns=[
            'review', 'rating', 'date', 'bank', 'source'
        ])
        print(f"Saved {len(valid_df)} cleaned records to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return

    # Generate comprehensive summary
    def generate_report(df):
        report = {
            'total_reviews': len(df),
            'banks': sorted(df['bank'].unique()),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
                'days': (datetime.strptime(df['date'].max(), '%Y-%m-%d') - 
                        datetime.strptime(df['date'].min(), '%Y-%m-%d')).days
            },
            'reviews_by_bank': df['bank'].value_counts().to_dict(),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
            'missing_data': {
                'reviews': sum(df['review'] == '(No text)'),
                'ratings': sum(df['rating'] == 0),
                'banks': sum(df['bank'].isna()),
                'dates': sum(df['date'].isna())
            }
        }
        return report

    report = generate_report(valid_df)
    
    print("\n=== DATA REPORT ===")
    print(f"Total Valid Reviews: {report['total_reviews']}")
    print(f"Banks Included: {', '.join(report['banks'])}")
    print(f"Date Range: {report['date_range']['start']} to {report['date_range']['end']} ({report['date_range']['days']} days)")
    
    print("\nReviews by Bank:")
    for bank, count in report['reviews_by_bank'].items():
        print(f"- {bank}: {count} reviews")
    
    print("\nRating Distribution:")
    for rating, count in report['rating_distribution'].items():
        print(f"- {rating}-star: {count}")
    
    print("\nMissing Data Handling:")
    print(f"- Empty reviews: {report['missing_data']['reviews']}")
    print(f"- Zero ratings: {report['missing_data']['ratings']}")
    print(f"- Missing banks: {report['missing_data']['banks']}")
    print(f"- Invalid dates: {report['missing_data']['dates']}")

def main():
    # Configuration
    INPUT_CSV = "Ethiopian_bank_reviews.csv"  # Your input file
    OUTPUT_CSV = "Ethiopian_bank_reviews_cleaned.csv"  # Output file
    
    print("Starting comprehensive data processing...")
    clean_review_data(INPUT_CSV, OUTPUT_CSV)
    print("\nProcessing complete! Check the output file and report.")
    try:
        df = pd.read_csv(OUTPUT_CSV)
        print(f"Loaded {len(df)} raw records from {OUTPUT_CSV}")
        print(f"Data shape: {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading file: {e}")

if __name__ == "__main__":
    main()