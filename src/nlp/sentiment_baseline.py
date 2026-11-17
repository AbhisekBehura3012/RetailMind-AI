import os
import re
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

REVIEWS = [
    ("The quality is amazing, highly recommend this product!", 1),
    ("Terrible experience, it broke after one use.", 0),
    ("Five stars. Exactly what I was looking for. Will buy again.", 1),
    ("Waste of money. Customer service was awful and rude.", 0),
    ("Pretty decent for the price, works as expected.", 1),
    ("Do not buy this. The material feels cheap and toxic.", 0),
    ("Fast shipping and great build quality.", 1),
    ("Defective item arrived, returning it immediately.", 0),
    ("I love the color and the battery life is surprisingly good.", 1),
    ("Stopped working completely within a week. Frustrating.", 0)
]

def clean_text(text):
    text = text.lower()
    return re.sub(r'[^a-z\s]', '', text)

def train_baseline():
    print("[INFO] Loading and cleaning review data...")
    X = [clean_text(r[0]) for r in REVIEWS]
    y = [r[1] for r in REVIEWS]
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    print("[INFO] Training Classical Sentiment Baseline...")
    pipeline.fit(X, y)
    
    preds = pipeline.predict(X)
    print("\n[INFO] Classification Report:")
    print(classification_report(y, preds, target_names=["Negative", "Positive"]))
    
    os.makedirs("models", exist_ok=True)
    model_path = 'models/sentiment_baseline_v12.joblib'
    joblib.dump(pipeline, model_path)
    print(f"[SUCCESS] Review Processing Baseline saved to {model_path}")

if __name__ == '__main__':
    train_baseline()
