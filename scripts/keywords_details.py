import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rake_nltk import Rake
import re


def extract_keywords(df):
    """
    Extract keywords from reviews.
    """
    rake = Rake()
    keywords = []
    for review in df:
        rake.extract_keywords_from_text(review)
        top_phrases = rake.get_ranked_phrases()
        for phrase in top_phrases:
            if 4 <= len(phrase.split()) <= 6 and "bethesda" in phrase and "mod" in phrase:
                keywords.append(phrase)

    return keywords


if __name__ == "__main__":
    # load reviews
    data = pd.read_csv('steam data/Skyrim_sentiment.csv')
    pd.set_option('display.max_columns', 500)

    phrases = extract_keywords(data["review"][(data["topic"] == 'Company & Modding Community')
                                                       & (data["sentiment"] == 'Negative') & (data["votes_up"] > 0)])

    for phrase in phrases:
        print(phrase)
