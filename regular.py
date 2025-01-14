import csv
import os
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from collections import Counter
regular_file = ''

def process_csv(filename):
    global regular_file
    df = pd.read_csv(filename)
    df = df.dropna(subset=["review"])
    df['aspects_sentiment'] = df['review'].apply(sentiment_analysis_aspects)
    df['overall_sentiment'] = df['review'].apply(TBsentiment)

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Create the new output file name by appending '_with_sentiments'
    output_filename = f'{base_filename}_with_sentiments.csv'

    # Define the output path for saving the processed CSV
    output_path = os.path.join('output', output_filename)

    regular_file = output_path
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
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(positive_text)
    

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Positive Sentiment Reviews Word Cloud', fontsize=20,fontweight='bold')
    plt.show()

    wordcloud_image_path = os.path.join('output', new_file_name)
    plt.savefig(wordcloud_image_path)
    wordcloud.to_file(wordcloud_image_path)

    return new_file_name

def positive_frequency_graph():
    global file
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_positive_frequency_graph.png"
    data = pd.read_csv(file)

    positive = data[data['overall_sentiment'] == 'positive']
    positive_text = ' '.join(positive['review'].tolist())
    positive_text = positive_text.translate(str.maketrans('', '', string.punctuation))
    positive_text = ' '.join([word for word in positive_text.split() if word.lower()])

        # Count word frequencies
    word_freq = Counter(positive_text.split())

    # Create a DataFrame from the word frequency
    df_word_freq = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

    # Sort the DataFrame by word frequency
    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)

    # Display the 20 most common words using a bar plot also include hot to cold color map
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.bar(df_word_freq['Word'].head(20), df_word_freq['Frequency'].head(20), color=plt.cm.coolwarm(df_word_freq['Frequency'].head(20) / df_word_freq['Frequency'].head(20).max()))
    ax.set(title='Top 20 Most Common Words in Positive Reviews', xlabel='Word', ylabel='Frequency')
    #add the frequency on top of each bar
    for i, freq in enumerate(df_word_freq.head(20)['Frequency']):
        ax.text(i, freq, str(freq), ha='center', va='bottom')
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)

    return new_file_name

def negative_word_cloud():
    global file
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_negative_wordcloud.png"
    data = pd.read_csv(file)

    negative = data[data['overall_sentiment'] == 'negative']
    negative_text = ' '.join(negative['review'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(negative_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Negative Sentiment Reviews Word Cloud', fontsize=20,fontweight='bold')
    plt.show()

    
    wordcloud_image_path = os.path.join('output', new_file_name)
    plt.savefig(wordcloud_image_path)
    wordcloud.to_file(wordcloud_image_path)

    return new_file_name

def negative_frequency_graph():
    global file
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_negative_frequency_graph.png"
    data = pd.read_csv(file)

    negative = data[data['overall_sentiment'] == 'negative']
    negative_text = ' '.join(negative['review'].tolist())
    negative_text = negative_text.translate(str.maketrans('', '', string.punctuation))
    negative_text = ' '.join([word for word in negative_text.split() if word.lower()])

        # Count word frequencies
    word_freq = Counter(negative_text.split())

    # Create a DataFrame from the word frequency
    df_word_freq = pd.DataFrame(word_freq.items(), columns=['Word', 'Frequency'])

    # Sort the DataFrame by word frequency
    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False)

    # Display the 20 most common words using a bar plot also include hot to cold color map
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.bar(df_word_freq['Word'].head(20), df_word_freq['Frequency'].head(20), color=plt.cm.coolwarm(df_word_freq['Frequency'].head(20) / df_word_freq['Frequency'].head(20).max()))
    ax.set(title='Top 20 Most Common Words in Negative Reviews', xlabel='Word', ylabel='Frequency')
    #add the frequency on top of each bar
    for i, freq in enumerate(df_word_freq.head(20)['Frequency']):
        ax.text(i, freq, str(freq), ha='center', va='bottom')
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)

    return new_file_name