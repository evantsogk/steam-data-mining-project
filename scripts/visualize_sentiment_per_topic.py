import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation as LDA
sns.set_theme(style="whitegrid")


if __name__ == "__main__":
    # load reviews
    data = pd.read_csv('steam data/Skyrim_sentiment.csv')

    # create sentiment percentages per topic dataframe
    topic_names = data['topic'].unique()
    pos, neg, neut = [], [], []
    for topic in topic_names:
        counts = data[data['topic'] == topic]['sentiment'].value_counts()
        total = np.sum(counts)
        pos.append(counts[0] / total)
        neg.append(counts[1] / total)
        neut.append(counts[2] / total)

    sentiment_per_topic = pd.DataFrame({'Positive': pos, 'Neutral': neut, 'Negative': neg}, index=topic_names)

    # visualize
    ax = sentiment_per_topic.plot(kind='barh', figsize=(11, 6), color=['g', 'y', 'r'])

    plt.gca().invert_yaxis()
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.xlabel('% of reviews')
    plt.ylabel('Topic')
    plt.title('The Elder Scrolls V: Skyrim\n Sentiment Proportion by Topic')
    plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100) for x in plt.gca().get_xticks()])
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        ax.annotate(str(round(width * 100, 1)) + '%', xy=(left + width / 2, bottom + height / 2),
                    ha='center', va='center',  color='w', fontsize=11)
    plt.show()
