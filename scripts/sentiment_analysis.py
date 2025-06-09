import pandas as pd
from datetime import datetime
import re
from transformers import pipeline
from textblob import TextBlob
import numpy as np

def compute_sentiment(df, method='textblob'):
    """Compute sentiment scores using specified method"""
    if method == 'distilbert':
        # Load DistilBERT sentiment analysis model
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
        )
        
        def get_sentiment(text):
            if len(str(text).strip()) == 0:
                return {'label': 'NEUTRAL', 'score': 0}
            try:
                result = sentiment_analyzer(text[:512])[0]  # Truncate to 512 tokens
                return {'label': result['label'].upper(), 'score': result['score']}
            except:
                return {'label': 'NEUTRAL', 'score': 0}
        
        # Apply sentiment analysis
        sentiments = df['review_text'].apply(lambda x: get_sentiment(x))
        df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
        df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
        
    elif method == 'textblob':
        def get_textblob_sentiment(text):
            if len(str(text).strip()) == 0:
                return {'label': 'NEUTRAL', 'score': 0}
            analysis = TextBlob(str(text))
            polarity = analysis.sentiment.polarity
            
            if polarity > 0.1:
                label = 'POSITIVE'
            elif polarity < -0.1:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {'label': label, 'score': abs(polarity)}
        
        sentiments = df['review_text'].apply(lambda x: get_textblob_sentiment(x))
        df['sentiment_label'] = sentiments.apply(lambda x: x['label'])
        df['sentiment_score'] = sentiments.apply(lambda x: x['score'])
    
    return df

def aggregate_sentiment(df):
    """Aggregate sentiment by bank and rating"""
    if 'app_name' not in df.columns or 'rating' not in df.columns:
        print("Required columns (app_name, rating) not found for aggregation")
        return None
    
    # Create aggregation
    aggregation = {
        'sentiment_score': ['mean', 'count'],
        'sentiment_label': lambda x: x.value_counts().to_dict()
    }
    
    grouped = df.groupby(['app_name', 'rating']).agg(aggregation).reset_index()
    
    # Flatten multi-index columns
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    # Rename columns for clarity
    grouped = grouped.rename(columns={
        'sentiment_score_mean': 'avg_sentiment_score',
        'sentiment_score_count': 'review_count',
        'sentiment_label_<lambda>': 'sentiment_distribution'
    })
    
    return grouped

def analyze_reviews(input_file, output_file, sentiment_method='textblob'):
    """Main function to process reviews"""
    try:
        # Read and clean data
        df = pd.read_csv(input_file)
                
        # Compute sentiment
        df = compute_sentiment(df, method=sentiment_method)
        
        # Aggregate results
        aggregated = aggregate_sentiment(df)
        
        # Save results
        df.to_csv(output_file, index=False)
        agg_output_file = output_file.replace('.csv', '_aggregated.csv')
        aggregated.to_csv(agg_output_file, index=False)
        
        print(f"Successfully processed data. Saved to:")
        print(f"- Cleaned data with sentiment: {output_file}")
        print(f"- Aggregated results: {agg_output_file}")
        
        return df, aggregated
    
    #except Exception as e:
        #print(f"Error processing data: {e}")
        #return None, None

# 