# Twitter Sentiment Analysis
The project aimed to classify tweets into two categories: hateful messages and non-hateful messages using machine learning.

# What was followed:
1. Data analysis - the analysis_data file in the sentiment_analysis folder
    
    I read the data from the csv, through the "info" function I could see what exactly is in the file - I found 3 columns (id, label and tweet) with 31962 input values each. The labels are divided into 2 categories - 1 for hatred tweets (2242) and 0 for the others (29720). I noticed that the total hatred tweets in the .csv file represent 7.014% of the total tweets. I looked for the number of tweets that are unique = 29530 and the number of tweets that appear multiple times in the dataset = 694 and plotted a histogram to illustrate the number of occurrences of tweets by total number of tweets in the dataset. I consider that the number of questions that are not unique does not negatively influence subsequent training because their number is relatively small compared to to the total number of tweets and most of the questions are repeated a small number of times.

2. Data processing - the data_processing file in sentiment_analysis folder
    
    After the data analysis, the data processing and cleaning step follows. I have defined a function where the TweetTokenizer function from the nltk library is called, which detects emoji, tags and other specific elements of a tweet, and then special characters like ^\x00-\x7F and # are removed. In the same function, after removing the # character, we split the tags using the wordninja.split function. Besides these things, I made sure that all words are lowercase and removed all punctuation using token.isalnum().The next step was to detect the words with the highest frequency in the processed list and remove them because they would have skewed the training more in one direction. Next, I made a word lemmatization function, this brings the tokens to their base form and makes sure that this form exists in the dictionary. The process is quite precise and takes a longer time to execute but is necessary to improve the quality of the text.
   
3. Training - the training file in the sentiment_analysis folder

    The next step is to train the model that estimates whether a tweet is hatred or not. Using the previously processed data I divided the dataset and the corresponding labels into training and testing using the train_test_split function which takes into account the distribution of the data. The dataset we transformed into a sparse array using CountVectorize. I chose to train a model using the Support Vector Machine algorithm because the problem is a supervised classification, and this algorithm performs very well in classification problems. The chosen kernel is linear because it has increased speed and is good for data with many features, so I felt it was well suited to solving this problem. The C parameter that controls the cost of misclassifying a point on the training data was chosen for the first training as equal to 1. In addition to these parameters, the SVM was also given class_weight = 'balanced' because we know that the percentage of hatred tweets is only 7%, so the dataset is unbalanced.
   
    In the optimization stage, the first thing I did was to rerun the training with different values for the C parameter. After this step, the K-Fold Cross Validation algorithm was implemented for C=1, in which the dataset was divided into k=5 subsets. The results obtained for f1_macro were: 0.80996824, 0.80632025, 0.80141542, 0.81610551, 0.79694175, the mean of the 5 results being = 0.8061502337140343. The results are very similar meaning that there are no anomalies in the dataset.


###Dataset: https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech#train.csv
