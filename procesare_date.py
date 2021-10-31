import re

import pandas as pd

from nltk import TweetTokenizer
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
import wordninja


def pre_processing_twiiter(tweets):
    tw_list = []
    stop_words = set(stopwords.words('english'))
    for tweet in tweets:
        tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
        tweet = tweet.replace('#', '')
        tweet = tweet.lower()
        tw_tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        tweet = tw_tokenizer.tokenize(tweet)
        tw_list2 = []
        for token in tweet:
            if token.isalnum():
                token = wordninja.split(token)
                for t in token:
                    if t not in stop_words:
                        tw_list2.append(t)
        tw_list.append(tw_list2)
    return tw_list


def remove_freq_words(tweets):
    freq_words = pd.Series(tweets).value_counts()[:10]
    freq_words = list(freq_words.index)[0]

    tw_list = []
    for tweet in tweets:
        tw_list2 = []
        for token in tweet:
            if token not in freq_words:
                tw_list2.append(token)
        tw_list.append(tw_list2)
    return tw_list


def lemmatizer_tweets(tweets):
    lemmatizer = WordNetLemmatizer()
    tw_list = []
    for tweet in tweets:
        tw_list2 = []
        for token in tweet:
            lemmatizer_token = lemmatizer.lemmatize(token)
            tw_list2.append(lemmatizer_token)
        tw_list.append(tw_list2)
    return tw_list


def call_all_functions(tweets):
    tweets = pre_processing_twiiter(tweets)
    tweets = remove_freq_words(tweets)
    tweets = lemmatizer_tweets(tweets)

    return tweets
