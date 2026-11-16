import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

MODEL_DIR = "models"
SENTIMENT_MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_sentiment_v12.joblib")
TICKET_MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_ticket_v12.joblib")

def get_review_sentiment_data():
    samples = [
        ("Amazing quality. Exceeded expectations, works smoothly right out of the box.", 1),
        ("Fast shipping and great fit. Arrived early and fits true to size. Will order again.", 1),
        ("Solid value for the price. Performs just like high-end brands at half the cost.", 1),
        ("Absolutely love the wireless earbuds, crystal clear sound and long battery life.", 1),
        ("Customer support was super helpful and resolved my sizing question immediately.", 1),
        ("Terrible battery life. Stopped charging after two days. Complete waste of money.", 0),
        ("Broken on arrival. Screen was cracked and packaging was torn open.", 0),
        ("Worst customer experience. Item never arrived and refund was delayed for weeks.", 0),
        ("Cheap plastic material, snapped in half during first use. Do not buy.", 0),
        ("Not worth the price at all, stopped working right after the return window.", 0),
    ] * 40
    texts = [t for t, _ in samples]
    labels = [l for _, l in samples]
    return texts, labels

def get_ticket_routing_data():
    tickets = [
        ("I was charged twice on my credit card for order #4821.", "Billing & Refunds"),
        ("Still waiting on my refund after returning the jacket last week.", "Billing & Refunds"),
        ("Promo code did not apply at checkout, please adjust my invoice.", "Billing & Refunds"),
        ("Why is there an extra subscription fee on my monthly statement?", "Billing & Refunds"),
        ("Tracking says delivered yesterday but nothing arrived at my door.", "Shipping & Delivery"),
        ("Package has been stuck in transit at the regional hub for 6 days.", "Shipping & Delivery"),
        ("Can I change my delivery address before the courier dispatches?", "Shipping & Delivery"),
        ("Express shipping took over a week, very disappointed with courier.", "Shipping & Delivery"),
        ("The zipper broke the very first time I tried wearing the coat.", "Product Defect"),
        ("Headphones have loud static noise in the left ear cup.", "Product Defect"),
        ("Blender motor overheated and started smoking after 30 seconds.", "Product Defect"),
        ("Received a scratched display unit instead of a brand new phone.", "Product Defect"),
        ("Cannot reset my password, verification email never arrives.", "Account & Tech Support"),
        ("Mobile app crashes every time I open my saved wishlist.", "Account & Tech Support"),
        ("Two-factor authentication locked me out of my loyalty account.", "Account & Tech Support"),
        ("Website checkout button freezes when selecting saved cards.", "Account & Tech Support"),
    ] * 25
    texts = [t for t, _ in tickets]
    labels = [l for _, l in tickets]
    return texts, labels

def run_day12_nlp():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Train Sentiment Baseline (TF-IDF + Multinomial Naive Bayes)
    X_sent, y_sent = get_review_sentiment_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X_sent, y_sent, test_size=0.20, random_state=42, stratify=y_sent)

    sent_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    sent_pipe.fit(X_tr, y_tr)
    preds_sent = sent_pipe.predict(X_te)
    print(f"[DAY 12 SENTIMENT] Accuracy: {accuracy_score(y_te, preds_sent):.4f} | F1: {f1_score(y_te, preds_sent):.4f}")
    joblib.dump(sent_pipe, SENTIMENT_MODEL_PATH)

    # 2. Train Ticket Routing Baseline
    X_tkt, y_tkt = get_ticket_routing_data()
    Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X_tkt, y_tkt, test_size=0.20, random_state=42, stratify=y_tkt)

    ticket_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", MultinomialNB(alpha=0.5))
    ])
    ticket_pipe.fit(Xt_tr, yt_tr)
    preds_tkt = ticket_pipe.predict(Xt_te)
    print(f"[DAY 12 TICKET ROUTING] Accuracy: {accuracy_score(yt_te, preds_tkt):.4f} | Macro F1: {f1_score(yt_te, preds_tkt, average='macro'):.4f}")
    print(classification_report(yt_te, preds_tkt))
    joblib.dump(ticket_pipe, TICKET_MODEL_PATH)

    print(f"[SUCCESS] Saved Day 12 NLP models to {MODEL_DIR}/")

if __name__ == "__main__":
    run_day12_nlp()
