import pandas as pd
import sklearn
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from collections import defaultdict
import argparse

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_keywords(df, ngram_range=(1,3), max_features=100):
    """Extract significant keywords and n-grams using TF-IDF"""
    print("Extracting keywords using TF-IDF...")
    
    # Preprocess text
    def preprocess(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', str(text).lower())
        return text
    
    # Initialize TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words='english',
        preprocessor=preprocess
    )
    
    # Fit and transform
    tfidf_matrix = tfidf.fit_transform(df['review_text'])
    feature_names = tfidf.get_feature_names_out()
    
    # Get top keywords per bank
    bank_keywords = {}
    for bank in df['app_name'].unique():
        bank_indices = df[df['app_name'] == bank].index
        bank_matrix = tfidf_matrix[bank_indices]
        bank_scores = bank_matrix.sum(axis=0).A1
        top_indices = bank_scores.argsort()[::-1][:20]  # Top 20 per bank
        bank_keywords[bank] = [feature_names[i] for i in top_indices]
    
    return bank_keywords

def enhance_with_spacy(keywords_dict):
    """Enhance keywords with spaCy noun phrases and named entities"""
    print("Enhancing keywords with spaCy...")
    
    enhanced_keywords = {}
    for bank, keywords in keywords_dict.items():
        # Combine all keywords for this bank
        combined_text = " ".join(keywords)
        doc = nlp(combined_text)
        
        # Extract noun phrases
        noun_phrases = set([chunk.text.lower() for chunk in doc.noun_chunks])
        
        # Extract named entities
        entities = set([ent.text.lower() for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG']])
        
        # Combine all
        enhanced = set(keywords) | noun_phrases | entities
        enhanced_keywords[bank] = list(enhanced)
    
    return enhanced_keywords

def topic_modeling(df, n_topics=5):
    """Apply LDA topic modeling to identify broader themes"""
    print("Applying topic modeling...")
    
    # Initialize and fit LDA
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['review_text'])
    
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method='online'
    )
    lda.fit(tfidf_matrix)
    
    # Get top words for each topic
    feature_names = tfidf.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
        topics[f"Topic_{topic_idx}"] = top_features
    
    return topics

def group_into_themes(keywords_dict, topics=None):
    """Group keywords into overarching themes with documentation"""
    print("Grouping keywords into themes...")
    
    # Define theme templates (can be customized)
    theme_templates = {
        'Account Access': ['login', 'password', 'account', 'access', 'authentic', 'security'],
        'Transaction Issues': ['transfer', 'payment', 'transaction', 'failed', 'delay', 'process'],
        'User Experience': ['app', 'interface', 'design', 'experience', 'navigation', 'ui'],
        'Customer Support': ['support', 'service', 'response', 'representative', 'help', 'call'],
        'Features': ['feature', 'request', 'functionality', 'update', 'version', 'tool']
    }
    
    bank_themes = {}
    for bank, keywords in keywords_dict.items():
        theme_counts = {theme: 0 for theme in theme_templates}
        theme_keywords = {theme: [] for theme in theme_templates}
        
        # Count matches with theme templates
        for keyword in keywords:
            for theme, markers in theme_templates.items():
                if any(marker in keyword for marker in markers):
                    theme_counts[theme] += 1
                    theme_keywords[theme].append(keyword)
        
        # Select top 3-5 themes for this bank
        top_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        bank_themes[bank] = {
            'themes': [theme[0] for theme in top_themes],
            'theme_keywords': {k: v for k, v in theme_keywords.items() if v},
            'documentation': f"Themes for {bank} were determined by matching extracted keywords against common banking app categories. The top {len(top_themes)} themes were selected based on keyword frequency and relevance."
        }
    
    return bank_themes

def analyze_themes(df):
    """Complete thematic analysis workflow"""
    # Step 1: Extract keywords
    bank_keywords = extract_keywords(df)
    
    # Step 2: Enhance with spaCy
    enhanced_keywords = enhance_with_spacy(bank_keywords)
    
    # Step 3: Optional topic modeling
    topics = topic_modeling(df) if len(df) > 100 else None
    
    # Step 4: Group into themes
    bank_themes = group_into_themes(enhanced_keywords, topics)
    
    return bank_themes

def main():
    """Command-line interface for thematic analysis"""
    parser = argparse.ArgumentParser(description='Thematic analysis of bank app reviews')
    parser.add_argument('input_file', help='Path to cleaned CSV file with reviews')
    parser.add_argument('output_file', help='Path for JSON output file with themes')
    
    args = parser.parse_args()
    
    print("Bank Review Thematic Analysis")
    print("=" * 50)
    
    try:
        # Read cleaned data
        df = pd.read_csv(args.input_file)
        
        if df.empty:
            print("Error: Input file is empty!")
            return
        
        # Perform thematic analysis
        themes = analyze_themes(df)
        
        # Save results
        import json
        with open(args.output_file, 'w') as f:
            json.dump(themes, f, indent=2)
        
        print(f"\nSuccess! Themes saved to {args.output_file}")
        print("\nSample of identified themes:")
        for bank, data in list(themes.items())[:2]:  # Show first 2 banks as sample
            print(f"\nBank: {bank}")
            print(f"Themes: {', '.join(data['themes'])}")
            print("Example keywords:")
            for theme, keywords in list(data['theme_keywords'].items())[:3]:  # Show first 3 themes
                print(f"- {theme}: {', '.join(keywords[:3])}...")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == '__main__':
    main()