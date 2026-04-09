# app_flask.py
from flask import Flask, request, jsonify
import joblib
from dataframe import correct


app = Flask(__name__)

# Загружаем модели
model = joblib.load('model.pkl')
vectorizer = joblib.load('TfIdfVectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = correct(data.get('text', ''))
    
    text_tfidf = vectorizer.transform([text])
    prediction = model.predict(text_tfidf)[0]
    
    return jsonify({"prediction": str(prediction)})

@app.route('/')
def home():
    return {"status": "ok"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)