# Personalized Recommendation & Ranking Platform 🚀

An end-to-end Machine Learning recommendation system built to simulate a high-traffic production environment (e.g., Video/E-commerce feeds). This platform incorporates a dual-stage Retrieval and Learning-to-Rank (LTR) pipeline, Diversity/Freshness Re-ranking, a high-performance Serving API, and a Statistical A/B Testing Dashboard.

---

## 🏗️ Architecture

1. **Candidate Retrieval**: Dual-encoder user/item embeddings mapped into a high-dimensional vector space, indexed and queried via **FAISS** (Facebook AI Similarity Search) for sub-millisecond approximate nearest neighbor (ANN) retrieval. Supported by a secondary collaborative filtering and popularity fallback.
2. **Feature Engineering**: Batch processing pipeline transforming raw impressions and clicks into dense feature vectors (Historical CTR, Category Affinities, Content Freshness).
3. **Learning-to-Rank (LTR)**: A highly optimized **LightGBM** binary classification model trained on engineered tabular features to predict `P(click)` and score candidates.
4. **Re-Ranking**: Implementation of **Maximal Marginal Relevance (MMR)** to balance raw relevance scores against categorical and embedding diversity, mitigating filter bubbles.
5. **Serving Layer**: A **FastAPI** application acting as the inference layer, integrating stable MD5 user hashing for deterministic A/B variant assignment.
6. **Experimentation**: An offline traffic simulator that streams users through the API and logs impressions/clicks. Results are evaluated on a **Streamlit** dashboard calculating Statistical Significance ($p$-values) and Lift for key product metrics (CTR, Watch Time).

## 🚀 Key Results (A/B Test)

- **Control**: 50% of traffic routed to Popularity Baseline.
- **Treatment**: 50% of traffic routed to LightGBM Ranker + MMR.
- **Lift**: Treatment demonstrated a statistically significant ($p < 0.05$) **~40% lift in CTR** and **~20% lift in Average Watch Time**.

## 🛠️ Tech Stack
- **Languages**: Python 3.9+
- **Machine Learning**: LightGBM, FAISS, Scikit-Learn
- **Data Engineering**: Pandas, Numpy, Parquet
- **Backend/Serving**: FastAPI, Uvicorn
- **Visualization/Dashboard**: Streamlit, Plotly

## 📂 Repository Structure
```
NeuralRank-Engine/
├── api/                  # FastAPI inference endpoints
├── app/                  # Streamlit A/B Test Dashboard
├── data/                 # Raw/Processed datasets & feature matrices
├── models/               # Serialized LightGBM ranker
├── src/
│   ├── data/             # E-Commerce synthetic behavior simulator
│   ├── features/         # CTR and embedding feature pipelines
│   ├── ranking/          # LightGBM training logic
│   ├── reranking/        # MMR Diversity implementation
│   ├── retrieval/        # FAISS Index builder and inference
│   └── experiments/      # Traffic simulator and logger
└── pyproject.toml        # Dependencies
```

## 💻 Local Setup

1. **Clone & Environment**:
```bash
git clone git@github.com:Shyam-SN/NeuralRank-Engine.git
cd NeuralRank-Engine
python3 -m venv venv
source venv/bin/activate
./venv/bin/python3 -m pip install -e .
```

2. **Generate Data & Train**:
```bash
./venv/bin/python3 src/data/simulator.py
./venv/bin/python3 src/features/build_features.py
./venv/bin/python3 src/retrieval/retriever.py
./venv/bin/python3 src/ranking/ranker.py
```

3. **Simulate A/B Test & Launch Dashboard**:
```bash
./venv/bin/python3 src/experiments/simulate_ab_test.py
streamlit run app/streamlit_app.py
```
