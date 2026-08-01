import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import argparse
from sklearn.model_selection import train_test_split

class LightGBMRanker:
    def __init__(self):
        self.model = None
        self.feature_cols = []
        
    def prepare_training_data(self, interactions, user_features, item_features):
        print("Preparing training data...")
        # Join interactions with user and item features
        df = interactions.merge(user_features, on='user_id', how='left')
        df = df.merge(item_features, on='item_id', how='left')
        
        # Label: 1 if clicked, 0 if impression (we only use impressions and clicks)
        # Note: interactions dataframe has one row per event. 
        # A click is an event, but every click also has a preceding impression.
        # Let's aggregate to get (user, item) -> clicked (1 or 0)
        
        # Let's just group by user, item to get labels
        events = df.groupby(['user_id', 'item_id']).agg(
            clicked=('event_type', lambda x: int((x == 'click').any())),
            timestamp=('timestamp', 'min') # first impression time
        ).reset_index()
        
        # Re-join with features based on user/item
        # This is a bit simplified, but perfectly functional for this setup
        train_df = events.merge(user_features, on='user_id', how='left')
        train_df = train_df.merge(item_features, on='item_id', how='left')
        
        # Define feature columns
        self.feature_cols = [
            'activity_level', 'user_ctr',
            'base_appeal', 'historical_ctr',
            'impressions', 'clicks',
            'avg_watch_time_x', 'avg_watch_time_y'
        ]
        
        # Add category affinities dynamically
        cat_cols = [c for c in user_features.columns if c.startswith('affinity_')]
        self.feature_cols.extend(cat_cols)
        
        # Feature: user affinity for the SPECIFIC item category
        # One-hot encode item category
        cat_dummies = pd.get_dummies(train_df['category'], prefix='cat')
        train_df = pd.concat([train_df, cat_dummies], axis=1)
        self.feature_cols.extend(cat_dummies.columns.tolist())
        
        X = train_df[self.feature_cols].fillna(0)
        y = train_df['clicked']
        
        return X, y
        
    def train(self, X, y):
        print("Training LightGBM Ranker...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1
        }
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10)]
        )
        print("Training complete.")
        
    def save(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump({'model': self.model, 'features': self.feature_cols}, f)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactions', type=str, default='data/raw/interactions.parquet')
    parser.add_argument('--users', type=str, default='data/features/user_features.parquet')
    parser.add_argument('--items', type=str, default='data/features/item_features.parquet')
    parser.add_argument('--output', type=str, default='models/lgbm_ranker.pkl')
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    interactions = pd.read_parquet(args.interactions)
    user_features = pd.read_parquet(args.users)
    item_features = pd.read_parquet(args.items)
    
    ranker = LightGBMRanker()
    X, y = ranker.prepare_training_data(interactions, user_features, item_features)
    ranker.train(X, y)
    ranker.save(args.output)

if __name__ == '__main__':
    main()
