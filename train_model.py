import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from utils.preprocess import clean_text

warnings.filterwarnings("ignore")

DATASET_PATH = "dataset/news.csv"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CM_FIG_PATH = "static/confusion_matrix.png"


def load_dataset(path: str) -> pd.DataFrame:
    """
    news.csv dosyasini yukler ve temel dogrulama yapar.
    Gerekli sutunlar:
    - text
    - label (FAKE/REAL veya 0/1)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset bulunamadi: {path}")

    df = pd.read_csv(path)

    required_cols = {"text", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError("news.csv dosyasinda 'text' ve 'label' sutunlari olmali.")

    df = df.dropna(subset=["text", "label"]).copy()

    # Etiketleri standartlastir
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df["label"] = df["label"].replace(
        {"1": "REAL", "0": "FAKE", "TRUE": "REAL", "FALSE": "FAKE"}
    )

    df = df[df["label"].isin(["FAKE", "REAL"])].copy()

    if df.empty:
        raise ValueError("Gecerli veri bulunamadi. Label degerleri FAKE/REAL olmali.")

    return df


def train() -> None:
    print("1) Dataset yukleniyor...")
    df = load_dataset(DATASET_PATH)

    print("2) Metin temizleniyor...")
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3) TF-IDF vectorizer olusturuluyor...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("4) Logistic Regression egitiliyor...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    print("5) Performans hesaplanıyor...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["FAKE", "REAL"],
        yticklabels=["FAKE", "REAL"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    os.makedirs("static", exist_ok=True)
    plt.savefig(CM_FIG_PATH)
    plt.close()

    model_bundle = {"model": model, "accuracy": accuracy}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nModel kaydedildi: {MODEL_PATH}")
    print(f"Vectorizer kaydedildi: {VECTORIZER_PATH}")
    print(f"Confusion matrix kaydedildi: {CM_FIG_PATH}")


if __name__ == "__main__":
    try:
        train()
    except Exception as exc:
        print(f"Hata olustu: {exc}")
