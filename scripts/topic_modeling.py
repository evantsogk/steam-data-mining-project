import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

pd.set_option('display.max_columns', 500)

if __name__ == "__main__":
    # load reviews
    data = pd.read_csv('steam data/Skyrim_cleaned.csv')

    # voted_up counts before cleaning
    #print(data['voted_up'].value_counts())

    # create tf-idf vectors
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(data['cleaned_review'])

    # topic modeling
    num_topics = 4
    lda = LDA(n_components=num_topics, max_iter=100, random_state=0, n_jobs=5)
    topic_probs = lda.fit_transform(vectors)

    # print topics
    topic_arr = []

    def print_topics(model, vectorizer, top_n=15):
        for idx, topic in enumerate(model.components_):
            temps = []
            for i in topic.argsort()[:-top_n - 1:-1]:
                temps.append((vectorizer.get_feature_names()[i]))
            topic_arr.append(temps)

    print_topics(lda, vectorizer)
    rows = [f'Topic {i}' for i in range(0, num_topics)]
    tpi = pd.DataFrame(topic_arr)
    tpi.index = rows
    print(tpi.T)

    # assign topics
    topic_names = {0: 'Content & Authenticity',
                   1: 'Playthrough Experience',
                   2: 'World Details',
                   3: 'Company & Modding Community'}

    topics = np.argmax(topic_probs, 1)
    topics = [topic_names[topic] for topic in topics]

    # save dataframe
    data['topic'] = topics
    data.to_csv('steam data/Skyrim_topics.csv', index=False)
