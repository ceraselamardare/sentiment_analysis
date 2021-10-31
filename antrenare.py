import numpy as np
import pandas as pd

from procesare_date import call_all_functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score

df_train = pd.read_csv('train.csv')
tweets_train = df_train['tweet'].tolist()
labels = df_train['label'].tolist()

tweets = pd.Series(df_train['tweet'].tolist())
processed_tweets = call_all_functions(tweets)

corpus = []
for i in range(len(processed_tweets)):
    corpus.append(' '.join(processed_tweets[i]))

cv = CountVectorizer()
cv.fit(corpus)

X = cv.transform(corpus).toarray()
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3)

linear_model = SVC(kernel='linear', C=1, verbose=True, class_weight='balanced', probability=False)
print('fit')
linear_model.fit(X=X_train, y=y_train)

predictions = linear_model.predict(X_test)
print(classification_report(y_test, predictions))

cv_scores = cross_val_score(linear_model, X, labels, cv=5, scoring='f1_macro')
print(cv_scores)
print(np.mean(cv_scores))
