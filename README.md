# RetailMind AI — Enterprise Retail Intelligence Platform

## 35-Day Milestone Roadmap & Audit Trail

| Phase | Days | Focus Area | Status |
| :--- | :---: | :--- | :--- |
| **Phase 1** | Days 01–04 | Classical ML | Complete |
| **Phase 2** | Days 05–08 | Deep Learning & Vision | Complete |
| **Phase 3** | Days 09–20 | Time-Series & GenAI (Generation Track Complete at Day 17) | In Progress |
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

---

## Day 17: TTS Voice Replies & CLIP Multimodal Search (Generation Track Milestone)

| Capability | Model / Engine Architecture | Multimodal Interface | Output / Artifact |
| :--- | :--- | :--- | :--- |
| **Voice Support Assistant** (`TTSVoiceEngine`) | Hugging Face TTS (`facebook/mms-tts-eng`) + Pure-Python `wave` Synthesizer Fallback | Speaks grounded support chatbot responses for accessible voice UX | `assets/audio/support_voice_*.wav` |
| **Multimodal Catalog Search** (`CLIPMultimodalSearch`) | Hugging Face CLIP (`openai/clip-vit-base-patch32`) + Shared Semantic-Visual Projection | Unified cosine-similarity retrieval via `search_by_text()` and `search_by_image()` | Ranked SKU matches + similarity scores |

### Generation Track Summary (Days 14–17 Complete)
* **Text & Voice**: Grounded multi-turn support chatbot with review theme summarization and synthesized `.wav` voice replies.
* **Image, Variations & Video**: Prompt-templated hero marketing assets, strength-swept `img2img` seasonal variations, and interpolated promo clips.
* **Multimodal Retrieval**: Cross-modal text-to-product and image-to-product search over shared catalog embeddings.
