# Install the library if you haven't already
# !pip install nltk

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon (do this once)
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Sample text data for analysis
texts = [
    "I love this new phone! It's absolutely amazing.",
    "This movie was a total disappointment. It was boring.",
    "The weather is pretty neutral today.",
    "The restaurant's service was terrible, but the food was delicious.",
    "The product is fine, I guess.",
]

print("Performing sentiment analysis on the following texts:")
for text in texts:
    print(f"- {text}")

print("\n--- Sentiment Analysis Results ---")

# Analyze the sentiment for each text
for text in texts:
    sentiment_scores = analyzer.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    print(f"\nText: '{text}'")
    print(f"Scores: {sentiment_scores}")
    print(f"Overall Sentiment: {sentiment}")    