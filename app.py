import os
import pickle

from flask import Flask, render_template, request

from utils.preprocess import clean_text

app = Flask(__name__)

MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None
model_accuracy = None
load_error = None


def load_artifacts() -> None:
    """Model ve vectorizer dosyalarini yukle."""
    global model, vectorizer, model_accuracy, load_error
    try:
        if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError("Model dosyalari yok. Once train_model.py calistirin.")

        with open(MODEL_PATH, "rb") as f:
            model_bundle = pickle.load(f)

        with open(VECTORIZER_PATH, "rb") as f:
            loaded_vectorizer = pickle.load(f)

        model = model_bundle["model"]
        model_accuracy = float(model_bundle.get("accuracy", 0.0))
        vectorizer = loaded_vectorizer
        load_error = None

    except Exception as exc:
        load_error = str(exc)
        model = None
        vectorizer = None
        model_accuracy = None


load_artifacts()


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        model_loaded=(model is not None and vectorizer is not None),
        load_error=load_error,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return render_template(
            "result.html",
            error="Model yuklenemedi. Once train_model.py calistirin.",
            prediction=None,
            confidence=None,
            model_accuracy=None,
        )

    try:
        user_text = request.form.get("news_text", "").strip()

        # Basit input dogrulama
        if not user_text:
            return render_template(
                "result.html",
                error="Lutfen haber metni giriniz.",
                prediction=None,
                confidence=None,
                model_accuracy=round(model_accuracy * 100, 2) if model_accuracy else None,
            )

        if len(user_text) < 20:
            return render_template(
                "result.html",
                error="Metin cok kisa. En az 20 karakter giriniz.",
                prediction=None,
                confidence=None,
                model_accuracy=round(model_accuracy * 100, 2) if model_accuracy else None,
            )

        cleaned = clean_text(user_text)
        if not cleaned:
            return render_template(
                "result.html",
                error="Gecerli bir metin giriniz.",
                prediction=None,
                confidence=None,
                model_accuracy=round(model_accuracy * 100, 2) if model_accuracy else None,
            )

        text_vec = vectorizer.transform([cleaned])
        pred = model.predict(text_vec)[0]
        probs = model.predict_proba(text_vec)[0]
        confidence = float(max(probs) * 100)

        return render_template(
            "result.html",
            error=None,
            prediction=pred,
            confidence=round(confidence, 2),
            model_accuracy=round(model_accuracy * 100, 2) if model_accuracy else None,
        )

    except Exception as exc:
        return render_template(
            "result.html",
            error=f"Beklenmeyen bir hata olustu: {exc}",
            prediction=None,
            confidence=None,
            model_accuracy=round(model_accuracy * 100, 2) if model_accuracy else None,
        )


if __name__ == "__main__":
    app.run(debug=True)
