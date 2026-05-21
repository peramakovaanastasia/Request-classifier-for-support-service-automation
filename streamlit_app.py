import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="Классификатор обращений", page_icon="📨", layout="centered"
)

st.title("Классификатор обращений в службу поддержки")

st.markdown("""
    Введите текст обращения клиента.  
    Сервис определит категорию обращения и приоритет обработки.
    """)

user_input = st.text_area(
    "Текст обращения:",
    height=100,
    placeholder="Например: У меня не проходит оплата картой",
)

if st.button("Предсказать"):
    if not user_input.strip():
        st.warning("Введите текст обращения.")
    else:
        try:
            response = requests.post(API_URL, json={"text": user_input}, timeout=5)

            if response.status_code == 200:
                result = response.json()

                category = result.get("category", "Неизвестно")
                priority = result.get("priority", "Неизвестно")

                st.success(f"Категория: {category}")

                if priority == "Высокий":
                    st.error(f"Приоритет: {priority}")
                elif priority == "Средний":
                    st.warning(f"Приоритет: {priority}")
                else:
                    st.info(f"Приоритет: {priority}")

            else:
                st.error(f"Ошибка API: {response.status_code}")

                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("API-сервер не запущен. Сначала запустите app.py.")
        except requests.exceptions.Timeout:
            st.error("API-сервер не ответил вовремя.")
        except Exception as error:
            st.error(f"Ошибка: {error}")
