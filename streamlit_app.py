import streamlit as st
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="Классификатор обращений", layout="wide")
st.title("📬 Классификатор обращений в поддержку")

# ---------- NHL: последние матчи ----------
@st.cache_data(ttl=3600)
def get_nhl_games(days_back=3):
    try:
        start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"https://api-web.nhle.com/v1/schedule/{start}"
        resp = requests.get(url, timeout=10).json()
        games = []
        for day in resp.get('gameWeek', []):
            for g in day.get('games', []):
                if g.get('gameState') == 'OFF':
                    games.append({
                        'date': day['date'],
                        'away': g['awayTeam']['placeName']['default'],
                        'home': g['homeTeam']['placeName']['default'],
                        'away_score': g['awayTeam'].get('score', 0),
                        'home_score': g['homeTeam'].get('score', 0),
                    })
        games.sort(key=lambda x: x['date'], reverse=True)
        return games[:10]
    except Exception as e:
        st.sidebar.error(f"NHL error: {e}")
        return []

# ---------- NHL: турнирная таблица (все команды) ----------
@st.cache_data(ttl=1800)
def get_nhl_standings():
    try:
        url = "https://api-web.nhle.com/v1/standings/now"
        resp = requests.get(url, timeout=10).json()
        standings = []
        for team in resp.get('standings', []):
            standings.append({
                'rank': team.get('leagueSequence', 0),
                'name': team.get('teamName', {}).get('default', ''),
                'gp': team.get('gamesPlayed', 0),
                'w': team.get('wins', 0),
                'l': team.get('losses', 0),
                'otl': team.get('otLosses', 0),
                'pts': team.get('points', 0),
            })
        standings.sort(key=lambda x: x['rank'])
        return standings   # все команды
    except Exception as e:
        st.sidebar.error(f"Standings error: {e}")
        return []

# ---------- Боковая панель: NHL матчи ----------
with st.sidebar:
    st.header("🏒 NHL: последние матчи")
    games = get_nhl_games()
    if games:
        for g in games:
            try:
                d = datetime.strptime(g['date'], "%Y-%m-%d").strftime("%d.%m")
            except:
                d = g['date']
            st.markdown(f"**{d}**")
            # Убрали строку с @, оставили только счёт
            st.markdown(f"**{g['away']} {g['away_score']}:{g['home_score']} {g['home']}**")
            st.divider()
    else:
        st.info("Нет данных о матчах")

# ---------- Правая панель: турнирная таблица NHL (все) ----------
with st.sidebar:
    st.header("📊 NHL: таблица")
    standings = get_nhl_standings()
    if standings:
        table = "| # | Команда | И | О |\n|--|--------|----|---|\n"
        for t in standings:
            table += f"| {t['rank']} | {t['name']} | {t['gp']} | {t['pts']} |\n"
        st.markdown(table)
    else:
        st.info("Нет данных таблицы")

# ---------- Классификатор ----------
user_input = st.text_area("Текст обращения:", height=150, placeholder="Не проходит оплата картой")

if st.button("🔍 Предсказать"):
    if user_input.strip():
        with st.spinner("Обработка..."):
            try:
                r = requests.post("http://127.0.0.1:8000/predict",
                                  json={"text": user_input}, timeout=5)
                if r.status_code == 200:
                    data = r.json()
                    cat = data.get('category', 'Неизвестно')
                    pri = data.get('priority', 'Низкий')
                    col1, col2 = st.columns(2)
                    col1.success(f"📂 Категория: **{cat}**")
                    if pri == "Высокий":
                        col2.error(f"⚠️ Приоритет: **{pri}**")
                    elif pri == "Средний":
                        col2.warning(f"📌 Приоритет: **{pri}**")
                    else:
                        col2.info(f"ℹ️ Приоритет: **{pri}**")
                else:
                    st.error(f"Ошибка сервера {r.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Сервер не запущен (порт 8000)")
            except Exception as e:
                st.error(f"Ошибка: {e}")
    else:
        st.warning("Введите текст")
