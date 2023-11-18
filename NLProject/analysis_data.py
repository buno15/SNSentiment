import pandas as pd
import praw
import stanza
from flair.models import TextClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

# # データの前処理
# stemmer = SnowballStemmer('english')
# stop_words = set(stopwords.words('english'))
#
#
# def clean_text(text):
#     # Remove URL、特殊文字
#     text = re.sub(r'http\S+', '', text)
#     text = re.sub(r'\W', ' ', text)
#     text = text.lower()
#     text = ' '.join([word for word in text.split() if word not in stop_words])
#     text = ' '.join([stemmer.stem(word) for word in text.split()])
#     return text


# Auth Reddit API
# client_id = 'your client id'
# client_secret = 'your secret'
# user_agent = 'SampleNLP'
# user_id = 'your id'
# password = 'your password'


# PRAW options
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, user_id=user_id,
                     password=password)

# データ収集対象のサブレディットを指定
subreddit_name = 'all'

# サブレディットからホット投稿を取得
subreddit = reddit.subreddit(subreddit_name)
posts = []

# Redditから投稿をanalysis
for post in subreddit.hot(limit=10000):
    clean_title = post.title
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(clean_title)
    sentimentNLTK = score['compound']

    analysis = TextBlob(clean_title)
    sentimentBlob = analysis.sentiment.polarity

    # doc = nlp(post.title)
    #
    # for sentence in doc.sentences:
    #     print(sentence.sentiment)

    # sentence = Sentence(post.title)
    # classifier.predict(sentence)
    # label = sentence.labels[0]
    # sentiment = label.value  # 'POSITIVE' または 'NEGATIVE'
    # confidence = label.score  # 確信度
    #
    # # 確信度のしきい値を設定
    # threshold = 0.75  # しきい値は適宜調整
    #
    # # ラベルを数値に変換
    # if confidence < threshold:
    #     sentimentFlair = 0  # neutral
    # else:
    #     sentimentFlair = 0.5 if sentiment == 'POSITIVE' else -0.5
    #
    # sentimentFlair *= confidence

    posts.append([clean_title, sentimentNLTK, sentimentBlob])

# Convert DataFrame
posts_df = pd.DataFrame(posts, columns=['title', 'sentimentNLTK', 'sentimentBlob'])

# output csv
posts_df.to_csv('reddit_posts.csv', index=False)
