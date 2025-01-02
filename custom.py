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
    

def process_aspect_csv(filename):
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

import ast  # Import the safe literal_eval function

def extract_aspect_data(data):
    aspect_data = []
    for index, row in data.iterrows():
        try:
            # Use literal_eval for safety
            aspect_sentiments = ast.literal_eval(row['aspects_sentiment'])  # Convert string to list of dictionaries
            for aspect_entry in aspect_sentiments:
                aspect_data.append({
                    'aspect': aspect_entry['aspect'],
                    'polarity': aspect_entry['polarity'],
                    'subjectivity': aspect_entry['subjectivity'],
                    'aspect_text': aspect_entry['aspect_text'],
                    'original_review': row['review'],
                    'date': row['date'],
                    'rating': row['rating']
                })
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing aspect_sentiment for row {index}: {e}")
            continue  # Skip this row if there's an error
    return pd.DataFrame(aspect_data)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

# Function to plot the density plot
def density_plot():
    global file
    data = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_aspect_density_plot.png"
    
    # Use extract_aspect_data to process the data
    aspect_df = extract_aspect_data(data)  # This will give you the aspect data
    
    # Use the 'polarity' as the 'compound' sentiment score for the density plot
    aspect_df['compound'] = aspect_df['polarity']  # If you want to use polarity directly
    
    # Generate the density plot
    x_vals = np.linspace(-1, 1, 200)
    plt.figure(figsize=(10, 6))
    
    # Loop through the unique aspects and plot their density
    for aspect in aspect_df['aspect'].unique():
        aspect_data = aspect_df[aspect_df['aspect'] == aspect]['compound']
        if len(aspect_data) > 1:  # Only plot if there are more than one data point
            kde = gaussian_kde(aspect_data)  # Kernel density estimation
            plt.plot(x_vals, kde(x_vals), label=aspect, alpha=0.7)
    
    plt.title("Density of Compound Sentiment Scores by Aspect")
    plt.xlabel("Compound Sentiment Score (Polarity from Aspect Sentiment)")
    plt.ylabel("Density")
    plt.xlim(-1, 1)
    plt.axvline(0, linestyle="--", color="black", alpha=0.6)
    plt.legend(title="Aspect")
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)
    return new_file_name

import seaborn as sns

# Function to plot box plot for sentiment distribution by aspect
def box_plot():
    data = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_aspect_box_plot.png"

    aspect_df = extract_aspect_data(data)
    aspect_df['compound'] = aspect_df['polarity']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=aspect_df, x="aspect", y="compound", palette="Set3")
    plt.title("Sentiment Score Distribution by Aspect")
    plt.xlabel("Aspect")
    plt.ylabel("Compound Sentiment Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)
    return new_file_name
