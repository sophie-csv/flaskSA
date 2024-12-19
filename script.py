import csv
import os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

def process_csv(filename):
    # Create a dictionary to group reviews by date
    date_reviews = defaultdict(list)

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Extract date and review from the CSV
            date = row['date']
            review = row['review']
            
            # Group reviews by date
            date_reviews[date].append(review)

    output_file = f'date_reviews_{str(datetime.now())}.csv'
    os.makedirs('output', exist_ok=True)  # Ensure the output directory exists
    with open(os.path.join('output', output_file), 'w') as f:
        writer = csv.writer(f)
        # Write header with aspects and sentiment included
        writer.writerow(['date', 'review', 'aspects', 'sentiments'])

        for date, reviews in date_reviews.items():
            for review in reviews:
                # Perform sentiment analysis for each review
                aspect_sentiments = sentiment_analysis_aspects(review)
                
                # Flatten the aspect-sentiment dictionary
                aspects = ', '.join(aspect_sentiments.keys())
                sentiments = ', '.join(aspect_sentiments.values())
                
                # Write the date, review, aspects, and sentiments as a row
                writer.writerow([date, review, aspects, sentiments])

    return output_file

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
