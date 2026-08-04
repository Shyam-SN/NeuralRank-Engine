from fastapi import FastAPI, HTTPException
import hashlib
import pandas as pd
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.retrieval.retriever import CandidateRetriever
from src.reranking.mmr import mmr_rerank

app = FastAPI(title="Recommender API", description="Personalized Recommender with A/B Testing")

# Global states (In a real app, these would be loaded from a DB/Cache)
class RecommendationService:
    def __init__(self):
        self.items_df = None
        self.users_df = None
        self.retriever = None
        self.ranker_model = None
        self.feature_cols = None
        
        self.load_models()
        
    def load_models(self):
        print("Loading data and models into memory...")
        self.items_df = pd.read_parquet('data/features/item_features.parquet')
        self.users_df = pd.read_parquet('data/features/user_features.parquet')
        
        self.retriever = CandidateRetriever(self.items_df, self.users_df)
        self.retriever.build_index()
        
        with open('models/lgbm_ranker.pkl', 'rb') as f:
            model_dict = pickle.load(f)
            self.ranker_model = model_dict['model']
            self.feature_cols = model_dict['features']
            
    def assign_variant(self, user_id):
        # Stable deterministic hashing for A/B assignment
        hash_val = int(hashlib.md5(f"user_{user_id}_salt2026".encode()).hexdigest(), 16)
        bucket = hash_val % 100
        if bucket < 50:
            return "control"
        else:
            return "treatment"
            
    def get_recommendations(self, user_id, k=20):
        variant = self.assign_variant(user_id)
        
        if variant == "control":
            # Baseline: Popular items only
            items = self.retriever.popular_items[:k]
            scores = [1.0 - (0.01 * i) for i in range(k)]
            return {"user_id": user_id, "variant": variant, "items": [{"item_id": i, "score": s} for i, s in zip(items, scores)]}
            
        else:
            # Treatment: ML Pipeline
            # 1. Retrieval
            candidates = self.retriever.retrieve(user_id, k=100)
            
            # 2. Feature Building
            user_features = self.users_df[self.users_df['user_id'] == user_id]
            if len(user_features) == 0:
                # Cold start
                return {"user_id": user_id, "variant": variant, "items": [{"item_id": i, "score": 1.0} for i in self.retriever.popular_items[:k]]}
                
            item_features = self.items_df[self.items_df['item_id'].isin(candidates)]
            
            # Cross join
            user_features = user_features.assign(key=1)
            item_features = item_features.assign(key=1)
            df = pd.merge(user_features, item_features, on='key').drop('key', axis=1)
            
            # One-hot encode category
            cat_dummies = pd.get_dummies(df['category'], prefix='cat')
            df = pd.concat([df, cat_dummies], axis=1)
            
            # Ensure all required columns exist
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0
                    
            X = df[self.feature_cols].fillna(0)
            
            # 3. Ranking
            scores = self.ranker_model.predict(X)
            
            # 4. MMR Re-ranking
            # We need embeddings for the candidates
            candidate_embs = [self.retriever.item_embeddings[self.retriever.item_id_map[i]] for i in df['item_id'].values]
            
            reranked_items = mmr_rerank(
                items=df['item_id'].tolist(),
                scores=scores.tolist(),
                embeddings=candidate_embs,
                lambda_param=0.7,
                top_n=k
            )
            
            return {"user_id": user_id, "variant": variant, "items": [{"item_id": i, "score": 1.0} for i in reranked_items]}

# Global instance
service = None

@app.on_event("startup")
def startup_event():
    global service
    service = RecommendationService()

@app.get("/recommendations")
def recommendations(user_id: int, k: int = 20):
    if service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return service.get_recommendations(user_id, k)
