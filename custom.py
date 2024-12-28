import pandas as pd
import numpy as np
from textblob import TextBlob
import os


# Pre-Defined
aspects = {
    "food": ["food", "taste", "flavor", "fries", "burger", "mac", "cheese", "salmon", "veggies", "chips", 
             "prepared", "pastry", "bread", "dessert", "meal", "dish", "menu", "portion", "delicious", 
             "fresh", "horrible", "old", "runny", "rosemary"],
    "service": ["service", "staff", "waiter", "waitress", "host", "manager", "employee", "friendly", 
                "courteous", "dedicated", "Amanda", "Jen", "seated", "kind", "treat", "welcome", 
                "helpful", "attentive", "slow", "rude"],
    "ambience": ["ambience", "atmosphere", "decor", "environment", "setting", "comfortable", 
                 "cozy", "noisy", "quiet", "clean", "music", "place", "vibe", "feel", "space"],
    "price": ["price", "cost", "expensive", "cheap", "value", "worth", "overpriced", "affordable", 
              "reasonable", "portion", "deal", "charges", "pricing"]
}

# Analyze sentiment
def get_aspect_sentiment(text):
    """
    Analyze sentiment for each aspect in the provided text.
    
    Args:
        text (str): The review text.
        
    Returns:
        list: A list of dictionaries containing sentiment scores for each aspect.
    """
    aspect_sentiments = []
    for aspect, keywords in aspects.items():
        # Extract sentences containing any of the aspect's keywords
        aspect_text = " ".join([sentence for sentence in text.split(".") 
                                if any(keyword.lower() in sentence.lower() for keyword in keywords)])
        if aspect_text:
            # Calculate sentiment using TextBlob
            blob = TextBlob(aspect_text)
            sentiment = blob.sentiment
            aspect_sentiments.append({
                "aspect": aspect,
                "polarity": sentiment.polarity,  # Sentiment polarity (-1 to 1)
                "subjectivity": sentiment.subjectivity,  # Sentiment subjectivity (0 to 1)
                "aspect_text": aspect_text  # The text associated with this aspect
            })
    return aspect_sentiments

def TBsentiment(review):
    analysis = TextBlob(review)
    if analysis.sentiment.polarity > 0:
        return 'positive'  # Positive sentiment
    elif analysis.sentiment.polarity < 0:
        return 'negative'  # Negative sentiment
    else:
        return 'neutral'  # Neutral sentiment
    

def process(filename):
    global file
    df = pd.read_csv(filename)
    df = df.dropna(subset=["review"])
    df['aspects_sentiment'] = df['review'].apply(get_aspect_sentiment)
    df['overall_sentiment'] = df['review'].apply(TBsentiment)

    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Create the new output file name by appending '_with_sentiments'
    output_filename = f'{base_filename}_with_aspect_sentiments.csv'

    # Define the output path for saving the processed CSV
    output_path = os.path.join('output', output_filename)

    file = output_path
    # Save the processed DataFrame as a new CSV file
    df.to_csv(output_path, index=False)

    return output_filename
