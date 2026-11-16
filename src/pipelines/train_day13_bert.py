import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

MODEL_NAME = "distilbert-base-uncased"
SENTIMENT_DIR = os.path.join("models", "bert_sentiment_v13")
TICKET_DIR = os.path.join("models", "bert_ticket_v13")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TICKET_LABELS = {
    0: "Billing & Refunds",
    1: "Shipping & Delivery",
    2: "Product Defect",
    3: "Account & Tech Support"
}

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=96):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

def load_amazon_sentiment_sample(n_samples=600):
    print("[INFO] Loading Amazon Polarity dataset from Hugging Face...")
    try:
        from datasets import load_dataset
        ds = load_dataset("amazon_polarity", split=f"train[:{n_samples}]")
        texts = [f"{t}. {c}" for t, c in zip(ds["title"], ds["content"])]
        labels = ds["label"]
        return texts, labels
    except Exception as e:
        print(f"[WARN] Could not fetch HF dataset ({e}). Using curated retail review fallback.")
        samples = [
            ("Amazing quality", "Exceeded expectations, works smoothly right out of the box.", 1),
            ("Terrible battery life", "Stopped charging after two days. Complete waste of money.", 0),
            ("Fast shipping and great fit", "Arrived early and fits true to size. Will order again.", 1),
            ("Broken on arrival", "Screen was cracked and packaging was torn open.", 0),
            ("Solid value for the price", "Performs just like high-end brands at half the cost.", 1),
            ("Worst customer experience", "Item never arrived and refund was delayed for weeks.", 0),
        ] * 50
        return [f"{t}. {c}" for t, c, _ in samples], [l for _, _, l in samples]

def create_ticket_routing_data():
    # Relabeled retail reviews & support inquiries across 4 operational queues
    base_tickets = [
        # 0: Billing & Refunds
        ("I was charged twice on my credit card for order #4821.", 0),
        ("Still waiting on my refund after returning the jacket last week.", 0),
        ("Promo code did not apply at checkout, please adjust my invoice.", 0),
        ("Why is there an extra subscription fee on my monthly statement?", 0),
        ("Need a tax receipt copy for my recent electronics purchase.", 0),
        # 1: Shipping & Delivery
        ("Tracking says delivered yesterday but nothing arrived at my door.", 1),
        ("Package has been stuck in transit at the regional hub for 6 days.", 1),
        ("Can I change my delivery address before the courier dispatches?", 1),
        ("Express shipping took over a week, very disappointed with courier.", 1),
        ("Received only half of my order, second box is missing.", 1),
        # 2: Product Defect
        ("The zipper broke the very first time I tried wearing the coat.", 2),
        ("Headphones have loud static noise in the left ear cup.", 2),
        ("Blender motor overheated and started smoking after 30 seconds.", 2),
        ("Received a scratched display unit instead of a brand new phone.", 2),
        ("Stitching came undone after one gentle wash cycle.", 2),
        # 3: Account & Tech Support
        ("Cannot reset my password, verification email never arrives.", 3),
        ("Mobile app crashes every time I open my saved wishlist.", 3),
        ("Two-factor authentication locked me out of my loyalty account.", 3),
        ("How do I pair the smart watch sensor with Android 15?", 3),
        ("Website checkout button freezes when selecting saved cards.", 3),
    ]
    # Expand dataset for batch training
    expanded = base_tickets * 15
    texts = [t for t, _ in expanded]
    labels = [l for _, l in expanded]
    return texts, labels

def fine_tune_classifier(texts, labels, num_labels, save_dir, task_name, epochs=2, batch_size=16):
    print(f"\n[INFO] Fine-tuning {MODEL_NAME} for {task_name} on {DEVICE}...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    model.to(DEVICE)

    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer)
    test_dataset = TextClassificationDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            batch_labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch + 1}/{epochs} | Train Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch["labels"].numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")
    print(f"\n[{task_name} EVAL] Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    print(classification_report(all_targets, all_preds, digits=4))

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[SUCCESS] Saved fine-tuned model & tokenizer to: {save_dir}")
    return acc, f1

def run_day13_pipeline():
    # 1. Sentiment Analysis (Replacing Day 12 TF-IDF Baseline)
    sent_texts, sent_labels = load_amazon_sentiment_sample(n_samples=600)
    fine_tune_classifier(
        sent_texts, sent_labels,
        num_labels=2,
        save_dir=SENTIMENT_DIR,
        task_name="Amazon Review Sentiment (BERT)",
        epochs=2
    )

    # 2. Support Ticket Routing (4 Operational Queues)
    ticket_texts, ticket_labels = create_ticket_routing_data()
    fine_tune_classifier(
        ticket_texts, ticket_labels,
        num_labels=4,
        save_dir=TICKET_DIR,
        task_name="Support Ticket Routing (BERT)",
        epochs=3
    )

    with open(os.path.join(TICKET_DIR, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(TICKET_LABELS, f, indent=2)

if __name__ == "__main__":
    run_day13_pipeline()
