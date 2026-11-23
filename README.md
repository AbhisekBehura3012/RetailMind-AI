# RetailMind AI ? Enterprise Retail Intelligence Platform

## 35-Day Milestone Roadmap & Audit Trail

| Phase | Days | Focus Area | Status |
| :--- | :---: | :--- | :--- |
| **Phase 1** | Days 01?04 | Classical ML | Complete |
| **Phase 2** | Days 05?08 | Deep Learning & Vision | Complete |
| **Phase 3** | Days 09?20 | Time-Series & GenAI | In Progress |
| **Phase 4** | Days 21?35 | MLOps | Upcoming |

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

---

## Day 16: Image Variations (Img2Img) & Short Promo Video Generation

| Capability | Diffusion / Motion Configuration | Control Parameters | Output Artifacts |
| :--- | :--- | :--- | :--- |
| **Catalog Image Variations** (`src/creative/variation_video_studio.py`) | `StableDiffusionImg2ImgPipeline` (`runwayml/stable-diffusion-v1-5`) + Deterministic SVG Renderer | Strength sweep (`0.35` Autumn, `0.55` Cyber Midnight, `0.70` Resort Daylight) & `guidance_scale` (`7.5?8.5`) | `assets/variations/RM-*_var_*.svg` & JSON manifests |
| **Short Promo Video Clip** (`generate_promo_clip`) | Keyframe Interpolation Engine (`16 frames @ 8 FPS = 2.0s`) | Smooth orbital lighting transitions, animated timeline progress, and fixed SKU seeds | `assets/promo/RM-*_promo_16f_8fps.svg` |

### Day 16 Architectural Highlights
* **Fidelity vs. Reinterpretation Sweep**: Sweeping denoising `strength` from `0.35` (65% original product structure preserved) to `0.70` (30% structure / 70% seasonal editorial reinterpretation) lets marketing teams generate campaign variants without losing SKU recognizability.
* **Unified Campaign Asset Packs**: `RetailMindPlatformService.create_campaign_pack()` bundles a hero visual, three themed `img2img` variations, and a 2-second promo clip in a single call.
