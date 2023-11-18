import my_model

lists = ['I like you.', 'I hate you.', 'I am a student.']

for text in lists:
    sample_vector = my_model.tfidf.transform([text])
    prediction = my_model.model.predict(sample_vector)

    print("Predicted Sentiment: \"" + text + "\" :", prediction[0])
