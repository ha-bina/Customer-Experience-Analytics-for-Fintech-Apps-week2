import pandas as pd
def clean_review_data(input_file, output_file):
    """
    Cleans and standardizes bank review data from a CSV file.
    
    Args:
        input_file (str): Path to raw CSV file
        output_file (str): Path to save cleaned CSV
    """
    # Load the raw data
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {len(df)} raw records from {input_file}")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Standardize column names (case-insensitive)
    column_mapping = {
        'review': ['review', 'text', 'comment', 'content'],
        'rating': ['rating', 'score', 'stars'],
        'date': ['date', 'review_date', 'timestamp'],
        'bank': ['bank', 'bank_name', 'app']
    }
    
    # Map columns to standard names
    for standard_name, alternatives in column_mapping.items():
        for alt in alternatives:
            if alt.lower() in [col.lower() for col in df.columns]:
                if standard_name not in df.columns:
                    df.rename(columns={alt: standard_name}, inplace=True)
                break

    # Ensure required columns exist
    missing_cols = [col for col in ['review', 'rating', 'date', 'bank'] if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Adding with default values.")
        for col in missing_cols:
            df[col] = pd.NA

    # Add source if missing
    if 'source' not in df.columns:
        df['source'] = 'Google Play'

    # Data cleaning pipeline
    def clean_data(df):
        # Remove duplicates
        df = df.drop_duplicates(subset=['review'], keep='first')
        
        # Handle missing values
        df['review'] = df['review'].fillna('(No text)')
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
        
        # Normalize dates
        df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Clean bank names
        if 'bank' in df.columns:
            df['bank'] = df['bank'].str.strip().replace({
                'cbe': 'CBE',
                'boa': 'BOA',
                'com.cbe.mobile.banking': 'CBE',
                'com.bankofabyssinia.mobilebanking': 'BOA',
                'com.dashen.mobilebanking': 'Dashen'
            })
        
        return df

    # Apply cleaning
    cleaned_df = clean_data(df.copy())
    
    # Save cleaned data
    try:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(cleaned_df)} cleaned records to {output_file}")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")
        return

    # Generate and display summary
    def generate_summary(df):
        summary = {
            'total_reviews': len(df),
            'reviews_per_bank': df['bank'].value_counts().to_dict(),
            'rating_distribution': df['rating'].value_counts().sort_index().to_dict(),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max()
            }
        }
        return summary

    summary = generate_summary(cleaned_df)
    
    print("\n=== Data Cleaning Summary ===")
    print(f"Total reviews after cleaning: {summary['total_reviews']}")
    print("\nReviews per bank:")
    for bank, count in summary['reviews_per_bank'].items():
        print(f"- {bank}: {count}")
    print("\nRating distribution:")
    for rating, count in summary['rating_distribution'].items():
        print(f"- {rating} star: {count}")
    print(f"\nDate range: {summary['date_range']['start']} to {summary['date_range']['end']}")

def main():
    # Configure input and output files
    input_csv = "Ethiopian_bank_reviews.csv"  # Change to your input file
    output_csv = "cleaned_Ethiopian_bank_reviews.csv"  # Change to desired output file
    
    print("Starting data cleaning process...")
    clean_review_data(input_csv, output_csv)
    print("\nData cleaning complete!")

if __name__ == "__main__":
    main()