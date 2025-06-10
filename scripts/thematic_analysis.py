import pandas as pd
import re
import spacy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict
import json
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Tokenize, clean, and lemmatize text"""
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    doc = nlp(text)
    
    # Lemmatization and stopword removal
    tokens = [
        token.lemma_.lower().strip() 
        for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and len(token.lemma_) > 2
    ]
    
    return " ".join(tokens)

def extract_keywords_tfidf(texts, ngram_range=(1,2), max_features=100):
    """Extract keywords using TF-IDF"""
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(texts)
    return tfidf.get_feature_names_out()

def extract_keywords_spacy(texts):
    """Extract keywords using spaCy"""
    keywords = set()
    for doc in nlp.pipe(texts, disable=["parser", "ner"]):
        for chunk in doc.noun_chunks:
            keywords.add(chunk.text.lower())
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and len(token.text) > 3:
                keywords.add(token.lemma_.lower())
    return list(keywords)

def cluster_keywords(keywords, n_clusters=5):
    """Cluster keywords into themes using K-means"""
    if not keywords:
        return {}
    
    # Create TF-IDF representation of keywords
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([" ".join(k.split()) for k in keywords])
    
    # Cluster keywords
    kmeans = KMeans(n_clusters=min(n_clusters, len(keywords)), random_state=42)
    kmeans.fit(X)
    
    # Organize keywords by cluster
    clusters = defaultdict(list)
    for keyword, label in zip(keywords, kmeans.labels_):
        clusters[f"Theme_{label+1}"].append(keyword)
    
    return dict(clusters)

def analyze_reviews(input_file, output_file):
    """Main analysis function"""
    # Read and preprocess data
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews")
    
    # Text preprocessing
    df['processed_text'] = df['review_text'].apply(preprocess_text)
    df = df[df['processed_text'].str.len() > 0]  # Remove empty texts
    
    # Sentiment analysis (using TextBlob as example)
    def get_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1: return ('POSITIVE', polarity)
        elif polarity < -0.1: return ('NEGATIVE', polarity)
        else: return ('NEUTRAL', polarity)
    
    df[['sentiment_label', 'sentiment_score']] = df['review_text'].apply(
        lambda x: pd.Series(get_sentiment(str(x)))
    )
    
    # Analyze by bank
    results = []
    theme_reports = {}
    
    for bank in df['app_name'].unique():
        bank_df = df[df['app_name'] == bank]
        texts = bank_df['processed_text'].tolist()
        
        # Extract keywords (using both methods)
        tfidf_keywords = extract_keywords_tfidf(texts)
        spacy_keywords = extract_keywords_spacy(texts)
        all_keywords = list(set(tfidf_keywords) | set(spacy_keywords))
        
        # Cluster keywords into themes
        themes = cluster_keywords(all_keywords, n_clusters=5)
        theme_reports[bank] = themes
        
        # Map reviews to themes
        def assign_theme(text):
            doc = nlp(text.lower())
            assigned = []
            for theme, keywords in themes.items():
                if any(keyword in text for keyword in keywords):
                    assigned.append(theme)
            return "|".join(assigned) if assigned else "General"
        
        bank_df['themes'] = bank_df['review_text'].apply(assign_theme)
        results.append(bank_df)
    
    # Combine all results
    final_df = pd.concat(results)
    
    # Save outputs
    final_df.to_csv(output_file, index=False)
    
    # Save theme reports
    #theme_file = output_file.replace('.csv', '_themes.json')
    #with open(theme_file, 'w') as f:
     #   json.dump(theme_reports, f, indent=2)
    
    #print(f"\nResults saved to:")
   # print(f"- Processed reviews: {output_file}")
    #print(f"- Theme analysis: {theme_file}")
    
    # Print sample output
    #print("\nSample output row:")
    #print(final_df.iloc[0][['app_name', 'sentiment_label', 'themes']])
    
   # print("\nSample theme clusters for first bank:")
    #first_bank = list(theme_reports.keys())[0]
    #for theme, keywords in list(theme_reports[first_bank].items())[:3]:
     #   print(f"{theme}: {', '.join(keywords[:5])}...")

if __name__ == '__main__':
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('input_file', help='Path to input CSV file')
    #parser.add_argument('output_file', help='Path for output CSV file')
    #args = parser.parse_args()
    
    analyze_reviews(input_file, output_file)