
import pandas as pd
from datetime import datetime
import re

def clean_csv(input_file, output_file):
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    # Define pattern: only English letters, spaces, and . , ! ? ' " -
    english_pattern = re.compile(r'^[a-zA-Z\s\.,!?\'"-]+$')

    if 'review_text' in df.columns:
        # Remove line breaks and extra spaces
        df['review_text'] = df['review_text'].astype(str).str.replace(r'[\n\r]', ' ', regex=True).str.strip()
        # Keep only rows where review_text matches the pattern
        mask = df['review_text'].apply(lambda x: bool(english_pattern.fullmatch(x)))
        df = df[mask]

    # Handle missing data
    # Fill missing app_name with 'Unknown'
    if 'app_name' in df.columns:
        df['app_name'] = df['app_name'].fillna('Unknown')
    
    # Fill missing review_text with empty string
    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].fillna('')
    
    # Fill missing rating with median or 0 if not available
    if 'rating' in df.columns:
        median_rating = df['rating'].median() if not df['rating'].isnull().all() else 0
        df['rating'] = df['rating'].fillna(median_rating)
    
    # Normalize dates to YYYY-MM-DD format
    if 'review_date' in df.columns:
        # Try to parse dates in various formats
        for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y']:
            try:
                df['review_date'] = pd.to_datetime(df['review_date'], format=date_format, errors='coerce')
                # If we successfully parsed any dates, break
                if not df['review_date'].isnull().all():
                    break
            except:
                continue
        
        # Convert to YYYY-MM-DD string format
        df['review_date'] = df['review_date'].dt.strftime('%Y-%m-%d')
        # Fill any remaining NaT (not a time) values with empty string
        df['review_date'] = df['review_date'].fillna('')

    # Select only the columns we want in the output
    output_columns = ['app_name', 'review_text', 'review_date', 'rating']
    # Keep only columns that exist in the dataframe
    output_columns = [col for col in output_columns if col in df.columns]
    
    # Create the output dataframe with only the desired columns
    cleaned_df = df[output_columns]
        # Save to CSV
    try:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Successfully cleaned and saved data to '{output_file}'")
    except Exception as e:
        print(f"Error saving CSV file: {e}")