# RetailMind AI — Enterprise Retail Intelligence Platform

## 🗺️ 35-Day Milestone Roadmap & Audit Trail
| Phase | Days | Focus Area | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Days 01–04 | Classical ML | ✅ Complete |
| **Phase 2** | Days 05–08 | Deep Learning & Vision | ✅ Complete |
| **Phase 3** | Days 09–20 | Time-Series & GenAI | 🔄 In Progress |
| **Phase 4** | Days 21–35 | MLOps | ⏳ Upcoming |

## 📊 Phase 3 Ledger
| Day | Focus | Key Deliverable | Status |
| :---: | :--- | :--- | :--- |
| **09** | RNN / LSTM | Demand Forecast | ✅ Complete |
| **10** | Seq2Seq | Multi-Horizon Forecast | ✅ Complete |
| **11** | **DCGAN** | **Synthetic Catalog Balancer** | **✅ Complete** |

---

## Day 13: Transformers & BERT — Text Intelligence (Sentiment & Ticket Routing)

| Task | Model Architecture | Accuracy | Macro F1 | Latency (CPU) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Review Sentiment (Day 12)** | TF-IDF + Logistic Regression | 0.8350 | 0.8342 | ~2 ms | Replaced Baseline |
| **Review Sentiment (Day 13)** | **Fine-Tuned DistilBERT (distilbert-base-uncased)** | **0.9167** | **0.9165** | ~45 ms | **Active Production** |
| **Ticket Routing (Day 13)** | **Fine-Tuned DistilBERT (4-Class Queue Router)** | **0.9667** | **0.9665** | ~48 ms | **Active Production** |

### Why BERT Replaced TF-IDF in RetailMind AI
* **Contextual Negation & Sarcasm**: TF-IDF treats *"not bad at all"* or *"refund never arrived"* as independent bag-of-words n-grams. Self-attention captures long-range negation and context across clauses (+8.1% accuracy lift).
* **Unified Multi-Task Backbone**: The same DistilBERT architecture handles both customer sentiment polarity and 4-way support ticket triage (Billing & Refunds, Shipping & Delivery, Product Defect, Account & Tech Support).
