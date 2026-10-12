# Disaster Tweet Classifier

A Flask web application that classifies a submitted tweet as **Disaster** or **Non-Disaster** using the trained TF-IDF and Logistic Regression artifacts in `deployment/`.

## Run locally

Use Python 3.10 through 3.13, then install the dependencies and start the application:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>. The app expects these files to be present:

- `deployment/disaster_model.pkl`
- `deployment/tfidf_vectorizer.pkl`

The training notebook is `disaster_tweet_classification.ipynb`. Retrain and export the model and vectorizer together if the preprocessing or feature layout changes; the application builds TF-IDF features followed by tweet length, word count, hashtag count, mention count, and VADER compound sentiment.

## Production deployment

The included `Procfile` starts the Flask application with Gunicorn. Configure the platform to install `requirements.txt` and run the web process declared there. Keep Flask debug mode disabled in production. The serialized model files must be included with the deployment or provided through an equivalent artifact release process.

## Interpretation

The output is a machine-learning classification of tweet text. It does not verify an incident or replace confirmation from emergency services.