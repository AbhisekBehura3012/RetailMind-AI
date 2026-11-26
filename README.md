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
| :--- | :--- | :--- | :--- |
| **09** | RNN / LSTM | Demand Forecast | Complete |
| **10** | Seq2Seq | Multi-Horizon Forecast | Complete |
| **11** | DCGAN | Synthetic Catalog Balancer | Complete |
| **12** | Classical NLP | TF-IDF + MultinomialNB Sentiment & Ticket Router | Complete |
| **14** | LLM Apps | Review Theme Summarizer & Grounded Support Chatbot | Complete |
| **15** | Diffusion | On-Brand Marketing Imagery Generator (`stable-diffusion-v1-5`) | Complete |
| **16** | Img2Img & Video | Catalog Image Variations Sweep & Short Promo Clip Generator | Complete |
| **17** | TTS & CLIP | Voice Support Replies & Multimodal (Text + Image) Catalog Search | Complete |
| **18** | RAG Foundations | Product & Policy Vector DB (`all-MiniLM-L6-v2` + Persisted Index) | Complete |

---

## Day 18: RAG Foundations — Vector DB & Semantic Knowledge Base

| Component | Configuration | Persistence & Metadata | Status |
| :--- | :--- | :--- | :--- |
| **Document Chunker** (`recursive_chunk_text`) | `RecursiveCharacterTextSplitter` (`chunk_size=260`, `chunk_overlap=50`) | Preserves `doc_id`, `doc_type` (`policy`, `faq`, `product`), `sku`, and `title` | Active |
| **Dense Vector Store** (`RetailVectorStore`) | `sentence-transformers/all-MiniLM-L6-v2` + `FAISS IndexFlatIP` (with L2-Normalized NumPy Fallback) | Persisted to `models/vector_db/chunks_metadata.json` & `vector_index.npz` | Active |

### Day 18 Architectural Highlights
* **Coherent Passage Chunking**: Overlapping sentence-aware chunks prevent policy conditions (such as return windows and warranty exceptions) from being split mid-clause.
* **Persistent Disk Index**: Pre-computed embeddings and chunk metadata are serialized under `models/vector_db/` so downstream RAG pipelines (Days 19–20) load instantaneously without re-embedding the catalog.
