import pandas as pd
import praw
import stanza
from flair.models import TextClassifier
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# 感情分析モデルのロード
# classifier = TextClassifier.load('en-sentiment')

# # Stanzaの英語モデルをダウンロード
# stanza.download('en')
#
# # NLPパイプラインの初期化
# nlp = stanza.Pipeline(lang='en')

# Auth Reddit API
client_id = 'your client id'
client_secret = 'your secret'
user_agent = 'SampleNLP'
user_id = 'your id'
password = 'your password'

# PRAW options
reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, user_id=user_id,
                     password=password)

# データ収集対象のサブレディットを指定
subreddit_name = 'all'  # 例としてPythonサブレディットを使用

# サブレディットからホット投稿を取得
subreddit = reddit.subreddit(subreddit_name)
posts = []

# Redditから投稿を収集
for post in subreddit.hot(limit=1000):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(post.title)
    sentimentNLTK = score['compound']

    analysis = TextBlob(post.title)
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

    posts.append([post.title, sentimentNLTK, sentimentBlob])

# DataFrameに変換
posts_df = pd.DataFrame(posts, columns=['title', 'sentimentNLTK', 'sentimentBlob'])

# データフレームをCSVファイルに保存
posts_df.to_csv('reddit_posts.csv', index=False)