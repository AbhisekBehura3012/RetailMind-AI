# RetailMind AI — Daily Development Log

## Day 1: End-to-End Baseline Pipeline & ML-as-a-Service
- **Changes Added**:
  - Initialized repo structure (data/, src/, models/, notebooks/, docs/).
  - Built scikit-learn Pipeline with ColumnTransformer (scaling + one-hot encoding).
  - Trained baseline Logistic Regression model and exported artefact via joblib.
  - Implemented service wrapper (src/service.py) for inference.
- **Business Value Added**:
  - Provides immediate baseline churn risk scoring for retail account teams.
- **Open Blockers / Points for Doubt-Clearing**:
  - Recall on churn is 53%, missing at-risk users due to class imbalance.

## Day 2: Decision Tree, Random Forest & SVM
- **Changes Added**:
  - Implemented multi-model comparative training harness (src/pipelines/train_comparative_models.py).
  - Evaluated Decision Tree, Random Forest, and SVM models on identical preprocessed features.
  - Selected Random Forest as Champion and persisted to models/churn_champion_v2.joblib.
  - Updated src/service.py to auto-resolve and serve Champion v2.
  - Added comparison table in README.md and created day02 notebook.
- **Business Value Added**:
  - Retention teams now capture ~75.7% of all churning accounts (up from 57.2%), preserving recurring revenue by triggering proactive retention outreach.
- **Open Blockers / Points for Doubt-Clearing**:
  - SVM training runtime is noticeably higher due to quadratic scaling; evaluate whether hyperparameter search or gradient boosting offers a better speed-to-recall trade-off.

## Day 3: Hyperparameter Tuning & Bias-Variance Diagnosis
- **Changes Added**:
  - Implemented 5-fold cross-validation grid search (\src/pipelines/train_day3_tuning.py\).
  - Tuned hyperparameter spaces for Logistic Regression, Random Forest, and SVM.
  - Diagnosed variance: default Random Forest showed a 12% train-validation gap; restricting depth to 6 reduced the gap to 3.5% without sacrificing recall.
  - Locked and saved tuned champion to \models/churn_v3.joblib\.
  - Added notebook \
otebooks/day03_tuning_bias_variance.ipynb\.
- **Business Value Added**:
  - The model generalises stably to unseen cohorts, preventing performance decay once deployed to live retail batches.
- **Open Blockers / Points for Doubt-Clearing**:
  - Model relies strictly on raw features without interaction terms (e.g. tenure vs charges ratio).

## Day 4: Feature Engineering, PCA & XGBoost Ensembles
- **Changes Added**:
  - Implemented \DomainFeatureEngineer\ in \src/pipelines/train_day4_final.py\.
  - Added PCA dimensionality reduction and compared Bagging (Random Forest) vs. Boosting (XGBoost).
  - Selected XGBoost with \scale_pos_weight\ as the final Phase 1 Production Model and saved to \models/churn_production_v4.joblib\.
  - Updated \src/service.py\ to auto-serve the Day 4 production model.
  - Added Day 4 notebook and updated the Phase 1 benchmark table in \README.md\.
- **Business Value Added**:
  - Reached **80.5% Churn Recall** and **0.8612 ROC-AUC**. Four out of five churning retail accounts are caught ahead of contract expiration, minimizing lost MRR.
- **Open Blockers / Points for Doubt-Clearing**:
  - Phase 1 complete. Ready for Phase 2: Deep Learning baseline (Keras ANN benchmark).

## Day 5: Deep Learning Propensity Model (Keras ANN)
- **Changes Added**:
  - Implemented 3-layer deep feed-forward neural network (\src/pipelines/train_day5_ann.py\) with ReLU activations, Batch Normalization, and Dropout.
  - Applied \EarlyStopping\ on validation loss to prevent tabular overfitting and configured class weights to counter the churn target imbalance.
  - Benchmarked ANN directly against the Phase 1 Classical Champion (XGBoost).
  - Saved model to \models/churn_ann_v5.keras\ and tabular preprocessor to \models/ann_preprocessor_v5.joblib\.
  - Updated \src/service.py\ to support dual-engine inference (classical pipeline or Keras deep learning model).
  - Documented engineering findings and 'when to use which' trade-offs in \README.md\.
- **Business Value Added**:
  - Proved empirically that while the ANN captures high recall (~78.3%), XGBoost remains the optimal production engine for tabular churn. This prevents over-engineering and keeps inference costs low while setting up the architecture for future multi-modal features.
- **Open Blockers / Points for Doubt-Clearing**:
  - Neural networks require careful feature scaling and struggle with mixed tabular types compared to tree-based splitters.

## Day 5: Deep Learning Propensity Model (Keras ANN)
- **Changes Added**:
  - Implemented 3-layer deep feed-forward neural network (\src/pipelines/train_day5_ann.py\) with ReLU activations, Batch Normalization, and Dropout.
  - Applied \EarlyStopping\ on validation loss to prevent tabular overfitting and configured class weights to counter the churn target imbalance.
  - Benchmarked ANN directly against the Phase 1 Classical Champion (XGBoost).
  - Saved model to \models/churn_ann_v5.keras\ and tabular preprocessor to \models/ann_preprocessor_v5.joblib\.
  - Updated \src/service.py\ to support dual-engine inference (classical pipeline or Keras deep learning model).
  - Documented engineering findings and 'when to use which' trade-offs in \README.md\.
- **Business Value Added**:
  - Proved empirically that while the ANN captures high recall (~78.3%), XGBoost remains the optimal production engine for tabular churn. This prevents over-engineering and keeps inference costs low while setting up the architecture for future multi-modal features.
- **Open Blockers / Points for Doubt-Clearing**:
  - Neural networks require careful feature scaling and struggle with mixed tabular types compared to tree-based splitters.

## Day 6: CNN Product Image Classification
- **Changes Added**:
  - Implemented standalone Vision pipeline (\src/pipelines/train_day6_cnn.py\) with Fashion-MNIST architecture prototyping.
  - Built 3-block augmented CNN for retail catalog auto-tagging.
  - Serialized model to \models/catalog_cnn_v6.keras\ and label mappings to \models/catalog_labels_v6.joblib\.
  - Added \CatalogVisionService\ to \src/service.py\ to support real-time product auto-tagging alongside churn prediction.
  - Added \
otebooks/day06_cnn_image_classification.ipynb\ and detailed documentation in \docs/phase2_deep_learning/day06_cnn_image_classification.md\.
- **Business Value Added**:
  - Cuts catalog onboarding latency by automating product category tagging directly from warehouse photos.
- **Open Blockers / Points for Doubt-Clearing**:
  - Training from scratch on small image crops limits fine-grained accuracy; ready for Day 7 Transfer Learning (pre-trained feature extractors) and object localization.

## Day 7: Transfer Learning & YOLO Object Detection
- **Changes Added**:
  - Built ResNet50 transfer learning pipeline in \src/vision/transfer_classifier.py\.
  - Added YOLO product detector in \src/vision/detector.py\.
  - Built catalog QA service in \src/vision/vision_service.py\.
  - Created notebook \
otebooks/day07_transfer_learning_yolo.ipynb\ and doc \docs/phase2_deep_learning/day07_transfer_learning_yolo.md\.
- **Business Value**:
  - Automatically flags non-compliant product photos and locates items for automated crop/thumbnail generation.

## Day 7: Deep Architecture Optimization & Batch Inference
- **Changes Added**:
  - Validated full-dataset batch evaluation against holdout split in inference service.
  - Profiled forward-pass CPU latency and memory footprint during batch scoring.
  - Updated README.md audit ledger with Day 7 evaluation benchmarks.
- **Business Value Added**:
  - Established rigorous full-cohort performance baselines to guarantee model stability before deploying automated retention triggers.
- **Open Blockers / Points for Doubt-Clearing**:
  - Assess whether embedding layers or transformer-based tabular architectures provide measurable gains over gradient boosting.

## Day 13: Transformers & BERT Text Intelligence
- **Changes Added**:
  - Built \src/pipelines/train_day13_bert.py\ to fine-tune \distilbert-base-uncased\ for Amazon review sentiment and 4-category support ticket routing.
  - Replaced Day 12's TF-IDF sentiment module in \src/service.py\ with \RetailMindTextService\.
  - Saved fine-tuned weights and tokenizers to \models/bert_sentiment_v13/\ and \models/bert_ticket_v13/\.
  - Created \
otebooks/day13_bert_text_intelligence.ipynb\ and updated \README.md\.
- **Business Value Added**:
  - Automatically routes incoming customer tickets into operational queues with >95% precision while flagging negative review spikes in real time.

## Day 12: Classical NLP Text Intelligence (TF-IDF + Multinomial Naive Bayes)
- **Changes Added**:
  - Built \src/pipelines/train_day12_nlp.py\ using \TfidfVectorizer(ngram_range=(1, 2))\ and \MultinomialNB\ for binary review sentiment and 4-queue support ticket routing.
  - Persisted serialized pipelines to \models/tfidf_sentiment_v12.joblib\ and \models/tfidf_ticket_v12.joblib\.
  - Updated \src/service.py\ with \RetailMindNLPService\ and added \
otebooks/day12_classical_nlp.ipynb\.

## Day 14: LLM Apps â€” Summaries & Support Chatbot
- Implemented \ReviewThemeSummarizer\ and \RetailSupportChatbot\ in \src/assistant/llm_assistant.py\ and exposed via \src/service.py\.

## Day 16: Image Variations & Short Promo Video Generation
- Built `src/creative/variation_video_studio.py` with `StableDiffusionImg2ImgPipeline` strength/guidance sweeps (`0.35` to `0.70`) and keyframe-interpolated short promo clip generation.
- Extended `src/service.py` with `create_campaign_pack()` and added `notebooks/day16_img2img_video_promo.ipynb`.
