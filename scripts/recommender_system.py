import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake


def extract_keywords(df, max_kwords):
    """
    Extract keywords from game description.
    """
    rake = Rake()
    keywords = df.shape[0] * [""]
    for i, description in enumerate(df['description']):
        rake.extract_keywords_from_text(description)

        # combine extracted keywords
        kword_list = []
        for phrase in rake.get_ranked_phrases():
            kword_list.extend(phrase.split())
        kword_list = list(dict.fromkeys(kword_list))  # ignore duplicates
        keywords[i] = " ".join(kword_list[:max_kwords])  # keep the most important words

    df['description'] = keywords
    return df


def combine_features(df):
    """
    Combine text features.
    """
    combined = df.shape[0] * [""]
    for i in range(df.shape[0]):
        # join names in one word and split multiple entities
        devs = df['developer'][i].replace(" ", "_").replace(';', " ")
        tags = df['steamspy_tags'][i].replace(" ", "_").replace(';', " ")
        desc = df['description'][i]
        combined[i] = devs + " " + tags + " " + desc
    return combined


def recommendations(data, cosine_sim, title, number_of_hits, min_ratings_dif):
    """
    Find similar games.
    """
    # game index
    game_idx = data.index[data['name'] == title][0]

    # similarity scores
    scores = pd.Series(cosine_sim[game_idx]).sort_values(ascending=False)

    # find recommendations
    top_scores = []
    most_similar = []
    for i in range(1, scores.shape[0]):
        index = scores.index.values[i]
        # ignore if the difference of positive ratings - negative ratings is small for better recommendations
        if data['positive_ratings'][index] - data['negative_ratings'][index] > min_ratings_dif:
            top_scores.append(scores[index])
            most_similar.append(data['name'][index])

        # stop when enough recommendations are found
        if len(most_similar) == number_of_hits:
            break

    return pd.DataFrame({'Recommended': most_similar, 'Score': top_scores})


if __name__ == "__main__":
    # load game information
    info = pd.read_csv('steam data/steam.csv', usecols=['appid', 'name', 'developer', 'steamspy_tags', 'positive_ratings', 'negative_ratings'])
    # load game description data
    descriptions = pd.read_csv('steam data/steam_description_data.csv', usecols=['steam_appid', 'short_description'])
    descriptions.columns = ['appid', 'description']
    # merge the dataframes
    data = pd.merge(info, descriptions, on="appid")

    # extract keywords from game description
    data = extract_keywords(data, max_kwords=10)

    # combine features
    features = combine_features(data)

    # create bag of words representations
    vectorizer = CountVectorizer(binary=True)
    vectors = vectorizer.fit_transform(features)

    # create the cosine similarity matrix
    sim_matrix = cosine_similarity(vectors)

    # get recommendations
    #game = 'The Elder Scrolls V: Skyrim Special Edition'
    #game = 'The Elder Scrolls V: Skyrim'
    #game = 'Frostpunk'
    game = 'Path of Exile'
    #game = 'Rocket League®'
    #game = 'Age of Empires® III: Complete Collection'
    #game = 'Assassin’s Creed® Brotherhood'
    #game = 'The Elder Scrolls® Online'
    #game = 'Fallout 4'
    #game = 'Need For Speed: Hot Pursuit'

    recommended = recommendations(data, sim_matrix, game, number_of_hits=15, min_ratings_dif=1000)

    print(recommended)
