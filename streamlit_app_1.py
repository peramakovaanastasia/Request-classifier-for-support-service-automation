import joblib
import streamlit as st
from pathlib import Path

from preprocess import preprocess_text


MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("TfIdfVectorizer.pkl")
ENCODER_PATH = Path("label_encoder.pkl")

PRIORITY_MAP = {
    "Оплата": "Высокий",
    "Техническая ошибка": "Высокий",
    "Доставка": "Средний",
    "Возврат и обмен": "Средний",
    "Спам": "Низкий",
}


st.set_page_config(
    page_title="Классификатор обращений",
    page_icon="📨",
    layout="centered",
)


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, vectorizer, label_encoder


def predict_category(text: str):
    cleaned_text = preprocess_text(text)

    if not cleaned_text:
        raise ValueError("Текст пустой после preprocessing")

    model, vectorizer, label_encoder = load_model_files()

    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    category = label_encoder.inverse_transform([prediction])[0]
    priority = PRIORITY_MAP.get(category, "Средний")

    return category, priority


st.title("Классификатор обращений в службу поддержки")

st.markdown(
    """
Введите текст обращения клиента.  
Сервис определит категорию обращения и приоритет обработки.
"""
)

user_input = st.text_area(
    "Текст обращения:",
    height=120,
    placeholder="Например: У меня не проходит оплата картой",
)

if st.button("Предсказать"):
    if not user_input.strip():
        st.warning("Введите текст обращения.")
    else:
        try:
            category, priority = predict_category(user_input)

            st.success(f"Категория: {category}")

            if priority == "Высокий":
                st.error(f"Приоритет: {priority}")
            elif priority == "Средний":
                st.warning(f"Приоритет: {priority}")
            else:
                st.info(f"Приоритет: {priority}")

        except FileNotFoundError as error:
            st.error(f"Не найден файл модели: {error}")
        except Exception as error:
            st.error(f"Ошибка: {error}")