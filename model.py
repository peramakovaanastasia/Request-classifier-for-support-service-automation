import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "correct_dataset.csv"

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "TfIdfVectorizer.pkl"
ENCODER_PATH = "label_encoder.pkl"

CONFUSION_MATRIX_PATH = "confusion_matrix.csv"
MODEL_ERRORS_PATH = "model_errors.csv"
METRICS_SUMMARY_PATH = "metrics_summary.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["text", "category"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    return df.reset_index(drop=True)


def save_confusion_matrix(y_test, y_pred, class_names) -> None:
    matrix = confusion_matrix(y_test, y_pred)

    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.to_csv(CONFUSION_MATRIX_PATH, encoding="utf-8-sig")

    print("\nConfusion Matrix:")
    print(matrix_df)


def save_model_errors(x_test, y_test, y_pred, label_encoder) -> None:
    errors = []

    for text, true_label, predicted_label in zip(x_test, y_test, y_pred):
        if true_label != predicted_label:
            true_category = label_encoder.inverse_transform([true_label])[0]
            predicted_category = label_encoder.inverse_transform([predicted_label])[0]

            errors.append(
                {
                    "text": text,
                    "true_category": true_category,
                    "predicted_category": predicted_category,
                }
            )

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(MODEL_ERRORS_PATH, index=False, encoding="utf-8-sig")

    print("\nКоличество ошибок модели:", len(errors_df))

    if not errors_df.empty:
        print("\nПримеры ошибок:")
        print(errors_df.head(10))


def save_metrics_summary(
    train_accuracy,
    train_f1,
    test_accuracy,
    test_f1,
) -> None:
    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "train",
                "accuracy": train_accuracy,
                "f1_weighted": train_f1,
            },
            {
                "dataset": "test",
                "accuracy": test_accuracy,
                "f1_weighted": test_f1,
            },
        ]
    )

    metrics_df.to_csv(METRICS_SUMMARY_PATH, index=False, encoding="utf-8-sig")


def main() -> None:
    df = load_data()

    x = df["text"]
    y = df["category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
    )

    x_train_tf = vectorizer.fit_transform(x_train)
    x_test_tf = vectorizer.transform(x_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(x_train_tf, y_train)

    y_train_pred = model.predict(x_train_tf)
    y_test_pred = model.predict(x_test_tf)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="weighted")

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="weighted")

    print("Train Accuracy:", train_accuracy)
    print("Train F1 weighted:", train_f1)

    print("\nTest Accuracy:", test_accuracy)
    print("Test F1 weighted:", test_f1)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    save_confusion_matrix(
        y_test,
        y_test_pred,
        label_encoder.classes_,
    )

    save_model_errors(
        x_test,
        y_test,
        y_test_pred,
        label_encoder,
    )

    save_metrics_summary(
        train_accuracy,
        train_f1,
        test_accuracy,
        test_f1,
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print("\nМодель, векторизатор и encoder сохранены.")
    print(f"Confusion matrix сохранена в {CONFUSION_MATRIX_PATH}")
    print(f"Ошибки модели сохранены в {MODEL_ERRORS_PATH}")
    print(f"Метрики сохранены в {METRICS_SUMMARY_PATH}")
    print(df["category"].value_counts())

if __name__ == "__main__":
    main()
