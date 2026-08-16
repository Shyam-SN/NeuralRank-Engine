import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api.main import RecommendationService

def simulate_experiment(num_users=5000):
    print("Initializing Recommendation Service for A/B Simulation...")
    service = RecommendationService()
    
    users = service.users_df['user_id'].values[:num_users]
    
    logs = []
    
    print("Simulating traffic...")
    for user_id in tqdm(users):
        # 1. Fetch recommendations
        resp = service.get_recommendations(user_id, k=20)
        variant = resp['variant']
        items = [i['item_id'] for i in resp['items']]
        
        # 2. Simulate outcomes
        # Control: generic popularity, low relevance for niche users
        # Treatment: personalized, high relevance
        
        if variant == 'control':
            # Base CTR 5%
            ctr = 0.05 + np.random.normal(0, 0.01)
            # Base watch time
            watch_time = np.random.normal(40, 10) # 40 mins
        else:
            # Base CTR 7%
            ctr = 0.07 + np.random.normal(0, 0.01)
            watch_time = np.random.normal(48, 10) # 48 mins
            
        ctr = np.clip(ctr, 0, 1)
        watch_time = max(0, watch_time)
        
        # Did they click? (We'll just aggregate metrics directly for the dashboard)
        clicks = int(np.random.binomial(20, ctr))
        impressions = 20
        
        logs.append({
            'user_id': user_id,
            'variant': variant,
            'impressions': impressions,
            'clicks': clicks,
            'watch_time_mins': watch_time
        })
        
    logs_df = pd.DataFrame(logs)
    
    os.makedirs('data/processed', exist_ok=True)
    logs_df.to_parquet('data/processed/experiment_logs.parquet')
    print("Simulation complete! Logs saved to data/processed/experiment_logs.parquet")

if __name__ == '__main__':
    simulate_experiment()
