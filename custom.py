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
from scipy.stats import pearsonr

# Function to plot box plot for sentiment distribution by aspect
def box_plot():
    data = pd.read_csv(file)
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_aspect_box_plot.png"

    data = extract_aspect_data(data)
    data['compound'] = data['polarity']
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x="aspect", y="compound", palette="Set3")
    plt.title("Sentiment Score Distribution by Aspect")
    plt.xlabel("Aspect")
    plt.ylabel("Compound Sentiment Score")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)
    return new_file_name


# Plot 3 - Correlation between word count and compound sentiment
def correlation():
    global file
    data = pd.read_csv(file)
    data = data.dropna(subset=['review'])
    base_name = os.path.splitext(os.path.basename(file))[0]
    new_file_name = f"{base_name}_correlational_plot.png"

    data['word_count'] = data['review'].apply(lambda x: len(str(x).split()))

    aspect_df = extract_aspect_data(data)
    aspect_df['compound'] = aspect_df['polarity']
    aspect_df['word_count'] = data['review'].apply(lambda x: len(str(x).split()))

    aspect_df = aspect_df.dropna(subset=['word_count', 'compound'])
    aspect_df = aspect_df[~aspect_df['word_count'].isin([float('inf'), float('-inf')])]
    aspect_df = aspect_df[~aspect_df['compound'].isin([float('inf'), float('-inf')])]
    base_name = os.path.splitext(os.path.basename(file))[0]
    
    new_file_name = f"{base_name}_correlation_plot.png"

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=aspect_df, x="word_count", y="compound", alpha=0.5)
    sns.regplot(data=aspect_df, x="word_count", y="compound", scatter=False, color="blue")
    plt.title("Correlation between Word Count and Compound Sentiment")
    plt.xlabel("Word Count")
    plt.ylabel("Compound Sentiment Score")
    corr_coefficient, _ = pearsonr(aspect_df['word_count'], aspect_df['compound'])

# Display the correlation coefficient on the plot
    plt.text(0.05, 0.95, f'Correlation Coefficient: {corr_coefficient:.2f}', 
         transform=plt.gca().transAxes, fontsize=14, verticalalignment='top', horizontalalignment='left',
         bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=1'))
    plt.show()

    image_path = os.path.join('output', new_file_name)
    plt.savefig(image_path)
    return new_file_name

from wordcloud import WordCloud


#helper
def generate_wordcloud(words, title, colormap):
    if words:
        wc = WordCloud(background_color='white', max_words=100, colormap=colormap).generate(words)
        plt.imshow(wc, interpolation="bilinear")
        plt.title(title)
        plt.axis("off")

def wordclouds():
    global file
    data = pd.read_csv(file)
    data = data.dropna(subset=['review'])  # Ensure 'review' column exists in 'data'
    base_name = os.path.splitext(os.path.basename(file))[0]

    # Process the data to extract aspects (assuming 'extract_aspect_data' is working correctly)
    aspect_df = extract_aspect_data(data)
    
    # Check if the aspect_df and data have a common column (e.g., 'id') to merge on
    if 'id' in aspect_df.columns and 'id' in data.columns:
        aspect_df = aspect_df.merge(data[['id', 'review']], on='id', how='left')
    else:
        # If no 'id' column, try joining based on the index
        aspect_df['review'] = data['review']
    
    aspect_df['compound'] = aspect_df['polarity']

    # This function will generate wordclouds for all aspects, positive and negative
    pos_files = []
    neg_files = []

    # Loop through aspects and generate word clouds for positive and negative polarity
    for aspect in aspects.keys():
        positive_words = " ".join(
            word for text in aspect_df[(aspect_df['aspect'] == aspect) & (aspect_df['compound'] > 0)]['review']
            if isinstance(text, str)  # Add this check to ensure text is a string
            for word in text.split()
        )
        negative_words = " ".join(
            word for text in aspect_df[(aspect_df['aspect'] == aspect) & (aspect_df['compound'] < 0)]['review']
            if isinstance(text, str)  # Add this check to ensure text is a string
            for word in text.split()
        )

        # Generate positive word cloud
        generate_wordcloud(positive_words, f"Positive Word Cloud for {aspect.capitalize()}", 'Blues')
        pos_file = f"{base_name}_positive_wordcloud_{aspect}.png"
        image_path_pos = os.path.join('wordclouds', pos_file)
        plt.savefig(image_path_pos)
        plt.close()

        # Generate negative word cloud
        generate_wordcloud(negative_words, f"Negative Word Cloud for {aspect.capitalize()}", 'Reds')
        neg_file = f"{base_name}_negative_wordcloud_{aspect}.png"
        image_path_neg = os.path.join('wordclouds', neg_file)
        plt.savefig(image_path_neg)
        plt.close()

        # Store the filenames
        pos_files.append(pos_file)
        neg_files.append(neg_file)

    return pos_files, neg_files