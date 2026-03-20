import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import pymorphy3
nltk.download('stopwords')

def correct(text):
    text = str(text).lower()
    text = re.sub(r'[^а-яa-z ]', '', text)
    words = text.split()
    stop = stopwords.words('russian')
    words = [w for w in words if w not in stop]
    words = [pymorphy3.MorphAnalyzer().parse(w)[0].normal_form for w in words]  # лемматизация
    return ' '.join(words)
    
data = pd.read_csv('dataset.csv')
data['text'] = data['text'].apply(correct)

data.to_csv('correct_dataset.csv', index=False, encoding='utf-8')