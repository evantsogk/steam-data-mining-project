import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import contractions
import re


stop_words = set(stopwords.words('english'))
stop_words.update(['game', 'skyrim', 'elder', 'scroll', 'steam', 'play', 'hour', 'time', 'day', 'year'])  # add some extra stopwords


def clean_text(text):
    """
    Clean text for bag of words.
    """
    text = text.lower()  # convert to lowercase
    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)  # remove urls
    text = contractions.fix(text)  # e.g. don't -> do not, you're -> you are

    # tokenize
    tokenizer = RegexpTokenizer(pattern='[a-z]+')  # keep only lowercase letters
    text = tokenizer.tokenize(text)

    # keep word lemmas
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(w) for w in text]

    # pos tagger
    tags = pos_tag(text)
    text = [word for word, pos in tags if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

    # remove stopwords
    text = [w for w in text if w not in stop_words and len(w) > 2]

    return text


if __name__ == "__main__":
    # load reviews
    # 377160_Fallout4.csv
    data = pd.read_csv('steam data/72850_TheElderScrollsVSkyrim.csv', usecols=['review', 'voted_up', 'votes_up'])
    data['review'] = data['review'].astype(str)

    # voted_up counts before cleaning
    print(data['voted_up'].value_counts())

    # clean reviews
    cleaned_reviews = []
    drop_rows = []
    for i, review in enumerate(data['review']):
        cleaned = clean_text(review)
        if len(cleaned) > 2:  # keep review only if it has more than two tokens
            cleaned_reviews.append(" ".join(cleaned))
            #print(cleaned)
        else:
            drop_rows.append(i)

    # drop rows
    data.drop(drop_rows, inplace=True)

    # voted_up counts after cleaning
    print(data['voted_up'].value_counts())

    # save cleaned dataframe
    data['cleaned_review'] = cleaned_reviews
    data.to_csv('steam data/Skyrim_cleaned.csv', index=False)
