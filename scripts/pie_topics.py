import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")


if __name__ == "__main__":
    # load reviews
    data = pd.read_csv('steam data/Skyrim_sentiment.csv')

    # topic counts
    topic_counts = data['topic'].value_counts()
    df_counts = pd.DataFrame({'topic': topic_counts.index, 'count': topic_counts.values})
    print(df_counts)

    # plot chart
    ax = df_counts.plot(kind='pie', y='count', autopct='%1.1f%%', labels=df_counts['topic'], startangle=90, legend=False)
    plt.title('The Elder Scrolls V: Skyrim\n Topics Discussed in Reviews')
    plt.ylabel('')
    plt.show()
