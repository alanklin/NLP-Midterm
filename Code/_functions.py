import subprocess
import sys
import os
from pathlib import Path

# Function to install dependencies
def install_requirements(requirements_file="requirements.txt"):
    """Installs dependencies from a requirements.txt file."""
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
            print("All dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            sys.exit(1)
    else:
        print(f"Error: {requirements_file} not found!")
        sys.exit(1)

# Run dependency installation
install_requirements()


### Imports ###

import pandas as pd
import numpy as np
from pathlib import Path
import os 

import re
import spacy
from geotext import GeoText
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import pipeline

import spacymoji

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

### Functions ###

nlp = spacy.load("en_core_web_sm")
nlp1 = spacy.blank("en")
nlp1.add_pipe("emoji", first = True)

def clean_data(df):
    # @author TP
    """This function is used to remove irrelevant data before input to model"""
    relevant_data = df['comment'].dropna()
    return relevant_data[1:]
    
def preprocess_text(text):
    """Clean and preprocess text by removing URLs, special characters, and lowercasing."""
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s\U0001F600-\U0001F64F]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase

    doc = nlp1(text)
    preprocessed_text = " ".join([token.text if not token._.is_emoji else token._.emoji_desc for token in doc])
    return preprocessed_text

def extract_country(text):
    """Extract country mentions using GeoText and spaCy."""
    geo = GeoText(text)
    countries = set(geo.countries)  # Extract countries from text
    
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":  # Geographic location
            countries.add(ent.text)
    
    return list(countries) if countries else None

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER and return the compound score."""
    return sia.polarity_scores(text)['compound']

def get_sentiment_label(score):
    """Convert VADER compound score into sentiment labels."""
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
    