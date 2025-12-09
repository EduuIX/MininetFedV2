# util/server/clientSelection/scopefl.py

import numpy as np

class SCOPEFL:
    def __init__(self):
      self.alpha = 0.5 
      self.beta = 0.5
      self.selection_percentage = 0.3

    def select_trainers_for_round(self, trainer_list, metrics):
        if not metrics or not trainer_list:
            return []

        client_scores = []
        
        # --- Find D_max and H_max ---
        all_samples = [metrics.get(trainer_id, {}).get('num_samples', 0) for trainer_id in trainer_list]
        all_entropies = [metrics.get(trainer_id, {}).get('entropy', 0) for trainer_id in trainer_list]
        
        # --- THIS IS THE FIX ---
        # Get the maximum value, or 0 if the list is empty
        d_max_val = max(all_samples) if all_samples else 0
        h_max_val = max(all_entropies) if all_entropies else 0
        
        # Ensure d_max and h_max are not zero to prevent division errors
        d_max = d_max_val if d_max_val > 0 else 1
        h_max = h_max_val if h_max_val > 0 else 1
        # --- END OF FIX ---
        
        # --- Calculate Relevance Score for each client ---
        for trainer_id in trainer_list:
            client_metrics = metrics.get(trainer_id, {})
            d_i = client_metrics.get('num_samples', 0)
            h_i = client_metrics.get('entropy', 0)
            
            relevance_score = (self.alpha * (d_i / d_max)) + (self.beta * (h_i / h_max))
            client_scores.append((trainer_id, relevance_score))
            
        # --- Sort clients by score in descending order ---
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # --- Select the top clients based on the percentage ---
        num_to_select = int(len(trainer_list) * self.selection_percentage)
        if num_to_select < 1 and len(trainer_list) > 0:
            num_to_select = 1
            
        selected_clients = [client[0] for client in client_scores[:num_to_select]]
        
        return selected_clients