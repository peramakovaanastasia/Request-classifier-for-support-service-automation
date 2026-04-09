from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
import pymorphy3

# Однократная загрузка ресурсов
nltk.download('stopwords', quiet=True)
STOP_WORDS = set(stopwords.words('russian'))
MORPH = pymorphy3.MorphAnalyzer()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^а-яёa-z -]", "", text, flags=re.IGNORECASE)
    words = text.split()
    # Лемматизация без удаления стоп-слов и ограничений по длине
    lemmas = []
    for w in words:
        try:
            lemma = MORPH.parse(w)[0].normal_form
            lemmas.append(lemma)
        except:
            lemmas.append(w)
    return " ".join(lemmas)

# Загрузка компонентов модели
vectorizer = joblib.load("TfIdfVectorizer.pkl")
model = joblib.load("model.pkl")
encoder = joblib.load("label_encoder.pkl")

# Приоритет на основе категории
PRIORITY_MAP = {
    "Оплата": "Высокий",
    "Техническая ошибка": "Высокий",
    "Доставка": "Средний",
    "Возврат и обмен": "Средний",
    "Спам": "Низкий"
}

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    raw_text = data.get('text', '').strip()
    if not raw_text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        cleaned = preprocess_text(raw_text)
        print(f"RAW: {raw_text}")
        print(f"CLEANED: {cleaned}")

        if not cleaned:
            return jsonify({'category': 'Не определена', 'priority': 'Низкий'})

        X_input = vectorizer.transform([cleaned])
        pred_id = model.predict(X_input)[0]
        category = encoder.inverse_transform([pred_id])[0]
        priority = PRIORITY_MAP.get(category, "Средний")

        return jsonify({'category': category, 'priority': priority})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
