import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import my_model

lists = pd.read_csv("test_data.csv")
num = len(lists)
sentiment = []

for text in lists['sentence']:
    sample_vector = my_model.tfidf.transform([text])
    prediction = my_model.model.predict(sample_vector)

    print("Predicted Sentiment: \"" + text + "\" :", prediction[0])

    probabilities = my_model.model.predict_proba(sample_vector)
    predicted_class_index = list(my_model.model.classes_).index(prediction[0])
    predicted_class_probability = probabilities[0][predicted_class_index]
    print(f'Probability: {predicted_class_probability:.4f}')
    accuracy = my_model.model.score(my_model.X_test, my_model.Y_test)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    sentimentNLTK = score['compound']
    
    analysis = TextBlob(text)
    sentimentBlob = analysis.sentiment.polarity
    
    sentiment.append((sentimentNLTK + sentimentBlob) / 2)
    
color = [('g' if s > 0 else 'r') for s in sentiment]


x = np.arange(1, num + 1)
y = np.array(sentiment)
plt.bar(x, y, color = color)
plt.xticks(np.arange(1, num + 1, 1))
plt.xlabel("Sentence number")
plt.ylabel("Sentimental value")
plt.show()