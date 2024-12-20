import csv
import os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal

file = ''

def process_csv(filename):
    global file
    df = pd.read_csv(filename)
    df = df.dropna(subset=["review"])
    df['aspects_sentiment'] = df['review'].apply(sentiment_analysis_aspects)
    df['overall_sentiment'] = df['review'].apply(TBsentiment)

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Create the new output file name by appending '_with_sentiments'
    output_filename = f'{base_filename}_with_sentiments.csv'

    # Define the output path for saving the processed CSV
    output_path = os.path.join('output', output_filename)

    file = output_path
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

import string 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from wordcloud import WordCloud

def positive_word_cloud():
    global file
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_positive_wordcloud.png"
    data = pd.read_csv(file)

    positive = data[data['overall_sentiment'] == 'positive']
    positive_text = ' '.join(positive['review'].tolist())
    positive_text = positive_text.translate(str.maketrans('', '', string.punctuation))
    positive_text = ' '.join([word for word in positive_text.split() if word.lower()])
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(positive_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Sentiment Reviews Word Cloud', fontsize=20,fontweight='bold')
    plt.show()

# NO TITLE FOR SOME REASON 
    wordcloud_image_path = os.path.join(os.path.join('output', 'media'), new_file_name)
    plt.savefig(wordcloud_image_path)
    wordcloud.to_file(wordcloud_image_path)

    return new_file_name