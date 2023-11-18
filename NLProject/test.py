from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

import my_model

lists = ['I like you.', 'I hate you.', 'I am a student.']

for text in lists:
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
    print("Sentimental Score: ", (sentimentNLTK + sentimentBlob) / 2, "\n")
