import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer


if __name__ == "__main__":
    # load reviews
    data = pd.read_csv('steam data/Skyrim_topics.csv')

    # print(data['topic'].value_counts())

    # sentiment analysis
    sentiment_tags = []
    vader = SentimentIntensityAnalyzer()

    for review in data['review']:
        compound = vader.polarity_scores(review)['compound']
        if compound > 0.1:
            sentiment_tags.append('Positive')
        elif compound < -0.1:
            sentiment_tags.append('Negative')
        else:
            sentiment_tags.append('Neutral')

    data['sentiment'] = sentiment_tags
    print(data['sentiment'].value_counts())

    # save dataframe
    data.to_csv('steam data/Skyrim_sentiment.csv', index=False)
