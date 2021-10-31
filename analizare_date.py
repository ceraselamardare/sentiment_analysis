import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize
from wordcloud import WordCloud

df_train = pd.read_csv('train.csv')
# info = df_train.info()
# print(info)


hatred_tweets = (df_train['label'] == 1).mean() * 100
print('Hatred tweets in train set = {}'.format(hatred_tweets))

tweets = pd.Series(df_train['tweet'].tolist())
unique_tweets = np.unique(tweets)

multiple_occuring_tweets = np.sum(tweets.value_counts() > 1)

print('Number of unique tweets = {}'.format(unique_tweets.shape[0]))
print('Number of multiple occuring tweets = {}'.format(multiple_occuring_tweets))

plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
plt.hist(tweets.value_counts(), bins=50)
plt.yscale('log')
plt.title('Distribution of unique and multiple occuring tweets')
plt.ylabel('Number of questions')
plt.xlabel('Number of occurrences of questions ')
plt.show

words = tweets.apply(lambda x: len(word_tokenize(x)))
print("Number of words = {}".format(words))

plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
plt.hist(words, bins=50, range=[0, 50], density=True, label='train')
plt.xlabel('Number of words')
plt.ylabel('Probability density')
plt.title('Distribution of words in data set')
plt.show()

cloud = WordCloud(width=1200, height=900).generate(" ".join(tweets.astype(str)))

plt.figure(figsize=(15, 10))
plt.style.use('ggplot')
plt.imshow(cloud)
plt.axis('Off')
plt.show()
