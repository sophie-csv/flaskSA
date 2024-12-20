import csv
import os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal


def process_csv(filename):
    df = pd.read_csv(filename)
    df = df.dropna(subset=["review"])
    df['aspects_sentiment'] = df['review'].apply(sentiment_analysis_aspects)
    df['overall_sentiment'] = df['review'].apply(TBsentiment)

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Create the new output file name by appending '_with_sentiments'
    output_filename = f'{base_filename}_with_sentiments.csv'

    # Define the output path for saving the processed CSV
    output_path = os.path.join('output', output_filename)

    # Save the processed DataFrame as a new CSV file
    df.to_csv(output_path, index=False)

    return output_filename

def TBsentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'  # Positive sentiment
    elif analysis.sentiment.polarity < 0:
        return 'negative'  # Negative sentiment
    else:
        return 'neutral'  # Neutral sentiment

import pandas as pd
import spacy
from textblob import TextBlob
nlp = spacy.load('en_core_web_sm')


def extract_aspects(review):
    doc = nlp(review)
    aspects = [token.text for token in doc if token.pos_ == 'NOUN']  # Extract nouns as potential aspects
    return aspects

# Function to calculate sentiment of specific aspects in a review
def sentiment_analysis_aspects(review):
    aspects = extract_aspects(review)
    aspect_sentiments = {}
    
    # Analyze the sentiment around each aspect
    for aspect in aspects:
        # We will extract sentences containing the aspect and analyze sentiment
        aspect_sentences = [sentence for sentence in review.split('.') if aspect in sentence]
        overall_sentiment = 0
        for sentence in aspect_sentences:
            analysis = TextBlob(sentence)
            overall_sentiment += analysis.sentiment.polarity
        
        # Determine if the sentiment is positive, negative, or neutral
        if overall_sentiment > 0:
            aspect_sentiments[aspect] = 'positive'
        elif overall_sentiment < 0:
            aspect_sentiments[aspect] = 'negative'
        else:
            aspect_sentiments[aspect] = 'neutral'
    
    return aspect_sentiments