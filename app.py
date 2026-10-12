
from flask import Flask, render_template, request
import os
import re
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# --------------------------------------------------
# Load the saved model and TF-IDF vectorizer
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "deployment")

model = joblib.load(
    os.path.join(MODEL_DIR, "disaster_model.pkl")
)

tfidf = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

analyzer = SentimentIntensityAnalyzer()


# --------------------------------------------------
# Text cleaning: same rules as your notebook
# --------------------------------------------------

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Generate the five extra features
# --------------------------------------------------

def create_extra_features(original_text):
    return pd.DataFrame([{
        "tweet_length": len(original_text),
        "word_count": len(original_text.split()),
        "hashtag_count": len(
            re.findall(r"#\w+", original_text)
        ),
        "mention_count": len(
            re.findall(r"@\w+", original_text)
        ),
        "sentiment_compound": analyzer.polarity_scores(
            original_text
        )["compound"]
    }])


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_tweet(tweet):
    cleaned_tweet = clean_text(tweet)

    if not cleaned_tweet:
        raise ValueError("Please enter a valid tweet.")

    # Convert cleaned text into TF-IDF features
    tfidf_features = tfidf.transform([cleaned_tweet])

    # Generate additional numeric features
    extra_features = create_extra_features(tweet)

    # Combine features in the same order as training
    extra_sparse = csr_matrix(
        extra_features[
            [
                "tweet_length",
                "word_count",
                "hashtag_count",
                "mention_count",
                "sentiment_compound"
            ]
        ].values
    )

    combined_features = hstack([
        tfidf_features,
        extra_sparse
    ]).tocsr()

    # Predict class: 0 = Non-disaster, 1 = Disaster
    prediction = int(model.predict(combined_features)[0])

    # Calculate model confidence if supported
    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(
            combined_features
        )[0]

        confidence = float(max(probabilities)) * 100

    if prediction == 1:
        label = "Disaster"
    else:
        label = "Non-Disaster"

    return label, confidence


# --------------------------------------------------
# Web page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    tweet = ""
    error = None

    if request.method == "POST":
        tweet = request.form.get("tweet", "").strip()

        try:
            if not tweet:
                raise ValueError("Please enter a tweet.")

            prediction, confidence = predict_tweet(tweet)

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Please check the model "
                "files and feature configuration."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        tweet=tweet,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=False)
