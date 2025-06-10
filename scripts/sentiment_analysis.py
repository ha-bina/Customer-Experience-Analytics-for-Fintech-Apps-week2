import pandas as pd
from datetime import datetime
import re
from transformers import pipeline
from textblob import TextBlob
import numpy as np

def compute_sentiment(df, method='textblob'):
    
    """Compute sentiment scores using specified method"""
    if method == 'distilbert':
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
            
            if polarity > 1.0 :
                label = 'POSITIVE'
            elif polarity < -1.0 :
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
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

def train_and_evaluate_nb(df, text_col='processed_review', label_col='label'):
    # Vectorize text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df[text_col])
    y = df[label_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    return clf, vectorizer
def get_top_keywords(df, text_col='processed_review', max_features=100):
    # Vectorize the dataset
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(df[text_col])

    # Get top keywords
    keywords = vectorizer.get_feature_names_out()
    print("Top Keywords:", keywords)
    return keywords, X, vectorizer

def extract_keywords_by_sentiment(df, sentiment_col='sentiment', text_col='processed_review', max_features=10):
    # Filter positive and negative reviews
    positive_reviews = df[df[sentiment_col] == 'positive'][text_col]
    negative_reviews = df[df[sentiment_col] == 'negative'][text_col]

    # Extract keywords from positive reviews
    vectorizer_pos = TfidfVectorizer(max_features=max_features)
    X_pos = vectorizer_pos.fit_transform(positive_reviews)
    print("Top Keywords in Positive Reviews:", vectorizer_pos.get_feature_names_out())

    # Extract keywords from negative reviews
    vectorizer_neg = TfidfVectorizer(max_features=max_features)
    X_neg = vectorizer_neg.fit_transform(negative_reviews)
    print("Top Keywords in Negative Reviews:", vectorizer_neg.get_feature_names_out())

    return {
        'positive_keywords': vectorizer_pos.get_feature_names_out(),
        'negative_keywords': vectorizer_neg.get_feature_names_out()
    }
def plot_positive_wordcloud(df, sentiment_col='sentiment', text_col='processed_review'):
    """Plot a word cloud for positive reviews."""
    positive_reviews = df[df[sentiment_col] == 'positive'][text_col]
    positive_text = ' '.join(positive_reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Positive Reviews')
    plt.show()
def plot_negative_wordcloud(df, sentiment_col='sentiment', text_col='processed_review'):
    """Plot a word cloud for negative reviews."""
    negative_reviews = df[df[sentiment_col] == 'negative'][text_col]
    negative_text = ' '.join(negative_reviews)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud for Negative Reviews')
    plt.show()
    