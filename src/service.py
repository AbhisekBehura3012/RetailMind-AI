import os
import joblib
import pandas as pd

SENTIMENT_MODEL_PATH = os.path.join("models", "tfidf_sentiment_v12.joblib")
TICKET_MODEL_PATH = os.path.join("models", "tfidf_ticket_v12.joblib")

class RetailMindNLPService:
    def __init__(self):
        if not os.path.exists(SENTIMENT_MODEL_PATH) or not os.path.exists(TICKET_MODEL_PATH):
            raise FileNotFoundError("Day 12 NLP models not found. Run src/pipelines/train_day12_nlp.py first.")
        self.sent_pipe = joblib.load(SENTIMENT_MODEL_PATH)
        self.ticket_pipe = joblib.load(TICKET_MODEL_PATH)

    def analyze_texts(self, texts: list[str]) -> pd.DataFrame:
        sent_preds = self.sent_pipe.predict(texts)
        sent_probs = self.sent_pipe.predict_proba(texts).max(axis=1)
        ticket_preds = self.ticket_pipe.predict(texts)
        ticket_probs = self.ticket_pipe.predict_proba(texts).max(axis=1)

        rows = []
        for idx, text in enumerate(texts):
            rows.append({
                "text": text[:50] + "..." if len(text) > 50 else text,
                "sentiment": "Positive" if sent_preds[idx] == 1 else "Negative",
                "sent_conf": round(float(sent_probs[idx]), 4),
                "routed_queue": ticket_preds[idx],
                "route_conf": round(float(ticket_probs[idx]), 4)
            })
        return pd.DataFrame(rows)

if __name__ == "__main__":
    samples = [
        "Absolutely love the wireless earbuds, crystal clear sound and fast delivery!",
        "My package says delivered but nothing arrived, and I was double billed.",
        "The blender motor started smoking on the first day of use.",
        "I cannot log into my account because the password reset link is broken."
    ]
    service = RetailMindNLPService()
    print("[ACTIVE SERVICE] Day 12 Classical NLP Intelligence (TF-IDF + Logistic Regression)")
    print(service.analyze_texts(samples).to_string(index=False))
