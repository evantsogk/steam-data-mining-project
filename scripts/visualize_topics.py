import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import pyLDAvis.sklearn
import pickle
sns.set_theme(style="whitegrid")


if __name__ == "__main__":
    # load lda model
    with open('steam data/lda.pkl', 'rb') as f:
        lda_model = pickle.load(f)

    # load reviews
    data = pd.read_csv('steam data/Skyrim_cleaned.csv')

    # create tf-idf vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['cleaned_review'])

    # visualize
    panel = pyLDAvis.sklearn.prepare(lda_model, vectors, vectorizer, mds='tsne')
    pyLDAvis.show(panel)

