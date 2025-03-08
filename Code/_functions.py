### Imports ###

import pandas as pd
import numpy as np
from pathlib import Path
import os 

import re
import spacy
from geotext import GeoText
import torch

import demoji
demoji.download_codes()

import statsmodels.api as sm
from plotnine import *

#from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
#from datasets import Dataset

#from transformers import pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

### Functions ###

nlp = spacy.load("en_core_web_sm")

def clean_data(df):
    # @author TP
    """This function is used to remove irrelevant data before input to model"""
    relevant_data = df['comment'].dropna()
    return relevant_data[1:]
    
def preprocess_text(text):
    """Clean and preprocess text by removing URLs, special characters, and lowercasing."""
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'\n', ' ', text)  # Remove new lines

    # remove emojis
    emojis = demoji.findall(text)
    for key in emojis.keys():
        text = text.replace(key, emojis[key].replace(":", " "))

    # NOTE : DO NOT ELIMINATE THE PUNCTUATION HERE
    text = text.lower().strip()  # Convert to lowercase
    return text

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

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download the VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER and return the compound score along with proportion scores."""
    scores = sia.polarity_scores(text)
    return {
        'compound': scores['compound'],
        'positive': scores['pos'],
        'neutral': scores['neu'],
        'negative': scores['neg']
    }
def get_sentiment_label(score):
    """Convert VADER compound score into sentiment labels."""
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'
    
    
def get_relative_path():
    """Get the relative path from Code/... to Data/vader_scores."""
    base_path = Path(__file__).resolve().parent
    target_path = base_path.parent / 'Data' / 'vader_scores'
    return target_path

