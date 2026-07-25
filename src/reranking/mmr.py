import numpy as np

def mmr_rerank(items, scores, embeddings, lambda_param=0.7, top_n=20):
    """
    Maximal Marginal Relevance Re-ranking.
    Lambda controls the trade-off: 
    1.0 means pure relevance (scores)
    0.0 means pure diversity
    """
    if not items or len(items) <= top_n:
        return items
        
    reranked_items = []
    selected_indices = []
    
    # Extract just the IDs and sort by initial score
    candidates = list(zip(items, scores, embeddings))
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize with the highest scoring item
    reranked_items.append(candidates[0][0])
    selected_indices.append(0)
    
    unselected = list(range(1, len(candidates)))
    
    while len(reranked_items) < top_n and unselected:
        best_mmr = -float('inf')
        best_idx = -1
        
        for idx in unselected:
            relevance = candidates[idx][1]
            
            # Calculate maximum similarity to already selected items
            emb_i = candidates[idx][2]
            max_sim = 0
            for sel_idx in selected_indices:
                emb_j = candidates[sel_idx][2]
                sim = np.dot(emb_i, emb_j)
                if sim > max_sim:
                    max_sim = sim
                    
            # MMR Score
            mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * max_sim)
            
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
                
        reranked_items.append(candidates[best_idx][0])
        selected_indices.append(best_idx)
        unselected.remove(best_idx)
        
    return reranked_items
