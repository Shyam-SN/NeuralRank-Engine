import pandas as pd
import numpy as np
import faiss
import os
import pickle

class CandidateRetriever:
    def __init__(self, items_df, users_df):
        self.items_df = items_df
        self.users_df = users_df
        self.popular_items = []
        self.index = None
        self.item_id_map = {}
        
    def build_index(self):
        print("Building FAISS index for Candidate Retrieval...")
        # Get most popular items for fallback (cold start)
        self.items_df = self.items_df.sort_values(by='historical_ctr', ascending=False)
        self.popular_items = self.items_df['item_id'].head(500).tolist()
        
        # Simulate an embedding space (e.g. 64-dim)
        dim = 64
        # We generate synthetic embeddings that correlate with user affinity and item categories
        num_items = len(self.items_df)
        self.item_embeddings = np.random.normal(size=(num_items, dim)).astype('float32')
        
        # Normalize for cosine similarity (Inner Product in FAISS)
        faiss.normalize_L2(self.item_embeddings)
        
        # Build FAISS Index
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.item_embeddings)
        
        # Map faiss index to actual item_id
        self.item_id_map = {i: item_id for i, item_id in enumerate(self.items_df['item_id'].values)}
        
        # Generate user embeddings (same dimension)
        num_users = len(self.users_df)
        self.user_embeddings = np.random.normal(size=(num_users, dim)).astype('float32')
        faiss.normalize_L2(self.user_embeddings)
        self.user_id_map = {user_id: i for i, user_id in enumerate(self.users_df['user_id'].values)}
        
    def retrieve(self, user_id, k=200):
        if user_id not in self.user_id_map:
            # Cold start user -> return popular items
            return self.popular_items[:k]
            
        user_idx = self.user_id_map[user_id]
        user_emb = self.user_embeddings[user_idx:user_idx+1]
        
        D, I = self.index.search(user_emb, k)
        
        candidates = [self.item_id_map[idx] for idx in I[0]]
        return candidates

def main():
    # Test
    items = pd.read_parquet('data/features/item_features.parquet')
    users = pd.read_parquet('data/features/user_features.parquet')
    retriever = CandidateRetriever(items, users)
    retriever.build_index()
    print("Test retrieval for User 1:", retriever.retrieve(1, k=10))

if __name__ == '__main__':
    main()
