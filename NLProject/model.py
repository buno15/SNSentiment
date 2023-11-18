import re

import pandas as pd
import praw
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# データの前処理
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # Remove URL、特殊文字
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\W', ' ', text)
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


df = pd.read_csv('reddit_posts.csv')

# TF-IDF Vector
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['title'])
df['average_score'] = df[['sentimentNLTK', 'sentimentBlob']].mean(axis=1)
Y = df['average_score'].apply(lambda x: 'neutral' if x == 0 else ('positive' if x > 0 else 'negative'))

# トレーニングセットとテストセットに分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Training model
model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# モデルの評価
predictions = model.predict(X_test)
print(classification_report(Y_test, predictions, zero_division=0))

list = ['I like you.', 'I hate you.', 'I am a student.']

for text in list:
    sample_vector = tfidf.transform([text])
    prediction = model.predict(sample_vector)

    print("Predicted Sentiment: \"" + text + "\" :", prediction[0])

# print("\n---Reddit Posts---")
#
# # PRAW options
# reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, user_id=user_id,
#                      password=password)
#
# # Redditからデータを収集
# subreddit = reddit.subreddit('all')
# top_posts = subreddit.hot(limit=10)
#
# # analysis recent posts
# for post in top_posts:
#     # URLと特殊文字を除去
#     processed_sample = clean_text(post.title)
#     sample_vector = tfidf.transform([processed_sample])
#     prediction = model.predict(sample_vector)
#
#     print("Predicted Sentiment: \"" + processed_sample + "\" :", prediction[0])
