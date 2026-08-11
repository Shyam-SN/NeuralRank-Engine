import pandas as pd
import numpy as np
import os
import argparse

def build_item_features(interactions_df, items_df):
    print("Building item features...")
    # Calculate item popularity and CTR
    item_stats = interactions_df.groupby('item_id').agg(
        impressions=('event_type', lambda x: (x == 'impression').sum()),
        clicks=('event_type', lambda x: (x == 'click').sum()),
        avg_watch_time=('watch_time', 'mean')
    ).reset_index()
    
    item_stats['historical_ctr'] = item_stats['clicks'] / (item_stats['impressions'] + 1e-5)
    
    features = pd.merge(items_df, item_stats, on='item_id', how='left')
    features['impressions'].fillna(0, inplace=True)
    features['clicks'].fillna(0, inplace=True)
    features['historical_ctr'].fillna(0, inplace=True)
    features['avg_watch_time'].fillna(0, inplace=True)
    
    return features

def build_user_features(interactions_df, users_df):
    print("Building user features...")
    user_stats = interactions_df.groupby('user_id').agg(
        total_impressions=('event_type', lambda x: (x == 'impression').sum()),
        total_clicks=('event_type', lambda x: (x == 'click').sum()),
        avg_watch_time=('watch_time', 'mean')
    ).reset_index()
    
    user_stats['user_ctr'] = user_stats['total_clicks'] / (user_stats['total_impressions'] + 1e-5)
    
    features = pd.merge(users_df, user_stats, on='user_id', how='left')
    features['total_impressions'].fillna(0, inplace=True)
    features['total_clicks'].fillna(0, inplace=True)
    features['user_ctr'].fillna(0, inplace=True)
    features['avg_watch_time'].fillna(0, inplace=True)
    
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/raw')
    parser.add_argument('--output_dir', type=str, default='data/features')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    interactions = pd.read_parquet(os.path.join(args.input_dir, 'interactions.parquet'))
    users = pd.read_parquet(os.path.join(args.input_dir, 'users.parquet'))
    items = pd.read_parquet(os.path.join(args.input_dir, 'items.parquet'))
    
    item_features = build_item_features(interactions, items)
    user_features = build_user_features(interactions, users)
    
    item_features.to_parquet(os.path.join(args.output_dir, 'item_features.parquet'))
    user_features.to_parquet(os.path.join(args.output_dir, 'user_features.parquet'))
    print("Feature building complete.")

if __name__ == '__main__':
    main()
