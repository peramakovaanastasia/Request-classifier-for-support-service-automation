import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('correct_dataset.csv')
df = df.dropna(subset=['text', 'category'])
df = df[df['text'].str.strip() != '']   # удаляем пустые строки

X = df['text'].tolist()
y_orig = df['category'].tolist()

encoder = LabelEncoder()
y = encoder.fit_transform(y_orig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, ngram_range=(1,2), sublinear_tf=True)
X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tf, y_train)

y_pred = model.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))

joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'TfIdfVectorizer.pkl')
joblib.dump(encoder, 'label_encoder.pkl')
print("\nМодели сохранены.")
