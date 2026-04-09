import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
df = pd.read_csv('correct_dataset.csv')
df.isnull().sum()
df = df.dropna(subset=['text', 'category'])
df.head()
X = df['text'].tolist()
y_orig = df['category'].tolist()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_orig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=1,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tf = vectorizer.fit_transform(X_train)
X_test_tf = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tf, y_train)
y_train_pred = model.predict(X_train_tf)
train_accuracy = accuracy_score(y_train_pred, y_train)
y_test_pred = model.predict(X_test_tf)
test_accuracy = accuracy_score(y_test_pred, y_test)
test_f1m = f1_score(y_test_pred, y_test, average='macro')
test_f1w = f1_score(y_test_pred, y_test, average='weighted')
print(f"\nОбучение (train):")
print(f"  Accuracy: {train_accuracy:.4f}")
print(f"\nТестовая выборка (test):")
print(f"  Accuracy: {test_accuracy:.4f}")
print(f"  F1-macro: {test_f1m:.4f}")
print(f"  F1-weighted: {test_f1w:.4f}")


joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'TfIdfVectorizer.pkl')