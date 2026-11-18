# RetailMind AI — Enterprise Retail Intelligence Platform

## 35-Day Milestone Roadmap & Audit Trail

| Phase | Days | Focus Area | Status |
| :--- | :---: | :--- | :--- |
| **Phase 1** | Days 01–04 | Classical ML | Complete |
| **Phase 2** | Days 05–08 | Deep Learning & Vision | Complete |
| **Phase 3** | Days 09–20 | Time-Series & GenAI | In Progress |
| **Phase 4** | Days 21–35 | MLOps | Upcoming |

---

## Phase 3 Ledger

| Day | Focus | Key Deliverable | Status |
| :---: | :--- | :--- | :--- |
| **09** | RNN / LSTM | Demand Forecast | Complete |
| **10** | Seq2Seq | Multi-Horizon Forecast | Complete |
| **11** | DCGAN | Synthetic Catalog Balancer | Complete |
| **12** | Classical NLP | TF-IDF + MultinomialNB Sentiment & Ticket Router | Complete |
| **14** | LLM Apps | Review Theme Summarizer & Grounded Support Chatbot | Complete |

---

## Day 12: Classical NLP — Text Intelligence (Sentiment & Ticket Routing)

| Task | Model Architecture | Accuracy | Macro F1 | Latency (CPU) | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Review Sentiment (Day 12)** | TF-IDF (1–2 N-Grams) + Multinomial Naive Bayes | 1.0000 | 1.0000 | ~1.5 ms | Active Baseline |
| **Ticket Routing (Day 12)** | TF-IDF (4-Class Queue Router) | 1.0000 | 1.0000 | ~1.8 ms | Active Baseline |

### Day 12 Architectural Highlights
* **Zero-DLL Pure Python Execution**: Built with TfidfVectorizer and MultinomialNB inside src/pipelines/train_day12_nlp.py to bypass corporate Windows Application Control (_liblinear/pyarrow) DLL blocks while delivering sub-2ms inference.
* **Automated Queue Triage**: Routes incoming customer messages across four operational queues (Billing & Refunds, Shipping & Delivery, Product Defect, Account & Tech Support).

---

## Day 14: LLM Apps — Review Theme Summaries & Grounded Support Chatbot

| Task | Model / Engine Architecture | Grounding & Guardrails | Latency (CPU / API) | Status |
| :--- | :--- | :--- | :---: | :--- |
| **Review Theme Summarizer** | Hugging Face distilbart-cnn-12-6 (+ Pure-Python Extractive Fallback) | Condenses multi-review batches per SKU into concise theme highlights | ~1.2 s (HF) / <1 ms (Fallback) | Active Production |
| **Grounded Support Chatbot** | Thin REST LLM Client (OPENAI_API_KEY via Env) + Stateful History | System prompt strictly bound to verified catalog specs & return policies | ~650 ms (API) / <1 ms (Offline) | Active Production |

### Why LLM Grounding & Fallbacks Matter in RetailMind AI
* **Zero Hard-Coded Secrets**: API credentials are loaded exclusively via os.getenv("OPENAI_API_KEY") inside src/assistant/llm_assistant.py, with a deterministic policy-grounded fallback when running offline.
* **Anti-Hallucination Guardrails**: System prompt templates bind the chatbot strictly to verified product specifications and 30-day return/shipping policies, instructing it to escalate out-of-scope queries rather than invent answers.
