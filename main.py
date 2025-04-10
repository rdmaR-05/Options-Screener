import feedparser
import time
from transformers import BertTokenizer, BertForSequenceClassification
from torch import softmax
import torch
import re

# Load FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('prosusAI/finbert')
model = BertForSequenceClassification.from_pretrained('prosusAI/finbert')

# Define function to analyze sentiment using FinBERT
def analyze_finbert_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=-1)
    sentiment = torch.argmax(probs, dim=-1).item()
    return sentiment, probs

# Define a function to dynamically extract stock names from the headline using patterns
def extract_stocks_from_headline(headline):
    # Simple regex to capture possible stock names (could be expanded based on tickers or specific patterns)
    # Example pattern: 'Reliance', 'Tata', 'HDFC', etc. (You can extend this list or refine pattern)
    patterns = ['Reliance', 'Tata', 'HDFC', 'Infosys', 'Wipro', 'ICICI', 'Bharti', 'Axis', 'Maruti', 'L&T']
    
    # Create a list to store detected stocks
    found_stocks = []
    
    for stock in patterns:
        if stock.lower() in headline.lower():
            found_stocks.append(stock)

    return found_stocks

# Function to fetch and process news headlines
def fetch_and_process_news():
    url = "https://www.cnbctv18.com/commonfeeds/v1/cne/rss/market.xml"  # CNBC RSS feed URL
    feed = feedparser.parse(url)

    # Initialize list for positive stock sentiment and Nifty sentiment
    good_stocks = []
    nifty_sentiment = None

    for entry in feed.entries:
        headline = entry.title
        sentiment, _ = analyze_finbert_sentiment(headline)

        # If the sentiment is positive, check for stocks mentioned in the headline
        if sentiment == 2:  # Positive sentiment
            found_stocks = extract_stocks_from_headline(headline)
            for stock in found_stocks:
                good_stocks.append(stock)

        # Track sentiment for Nifty-related headlines
        if "Nifty" in headline:
            nifty_sentiment = sentiment

    # Output the results
    if good_stocks:
        print("Good Stocks with Positive Sentiment: ", set(good_stocks))  # Remove duplicates using set
    else:
        print("No good stocks found with positive sentiment.")

    if nifty_sentiment is not None:
        print(f"Nifty Sentiment: {nifty_sentiment} (0: Negative, 1: Neutral, 2: Positive)")
    else:
        print("No Nifty-related headlines found.")

# Run the function every 10 minutes to get live sentiment, filter stocks, and track Nifty headlines
while True:
    fetch_and_process_news()
    time.sleep(600)  # Wait for 10 minutes before fetching the news again
