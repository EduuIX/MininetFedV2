# util/server/controller.py

import random
import numpy as np
import pandas as pd
from clientSelection import *
from aggregator import *
import importlib
from contextual_bandit import ContextualBandit

def criar_objeto(pacote, nome_classe):
    try:
        modulo = importlib.import_module(f"{pacote}")
        classe = getattr(modulo, nome_classe)
        return classe()
    except (ModuleNotFoundError, AttributeError) as e:
        print(f"Erro: {e}")
        return None

class Controller:
    def __init__(self, min_trainers=2, num_rounds=5, client_selector='Random', aggregator="FedAvg"):
        self.trainer_list = []
        self.min_trainers = min_trainers
        self.current_round = 0
        self.num_rounds = num_rounds
        self.num_responses = 0
        self.client_training_response = {}
        self.acc_list = []
        self.mean_acc_per_round = []
        self.clientSelection = criar_objeto("clientSelection", client_selector)
        self.aggregator = criar_objeto("aggregator", aggregator)
        self.metrics = {}
        self.global_model_weights = None

        pruning_rates = [0.2, 0.4, 0.6, 0.8]
        self.bandit = ContextualBandit(pruning_rates)
        self.client_pruning_rates = {}

    def get_trainer_list(self): return self.trainer_list
    def get_current_round(self): return self.current_round
    def get_num_trainers(self): return len(self.trainer_list)
    def get_num_responses(self): return self.num_responses
    def get_mean_acc(self):
        if not self.acc_list: return 0.0
        mean = float(np.mean(np.array(self.acc_list)))
        self.mean_acc_per_round.append(mean)
        return mean

    def update_metrics(self, trainer_id, metrics): self.metrics[trainer_id] = metrics
    def update_num_responses(self): self.num_responses += 1
    def reset_num_responses(self): self.num_responses = 0
    def reset_acc_list(self): self.acc_list = []
    def update_current_round(self): self.current_round += 1
    def add_trainer(self, trainer_id): self.trainer_list.append(trainer_id)
    def add_client_training_response(self, id, response): self.client_training_response[id] = response
    def add_accuracy(self, acc): self.acc_list.append(acc)

    def set_initial_global_model(self, weights):
        if self.global_model_weights is None:
            print("Received initial model from a client. Setting as global model.")
            self.global_model_weights = [np.asarray(w, dtype=np.float32) for w in weights]

    def select_trainers_for_round(self):
        return self.clientSelection.select_trainers_for_round(self.trainer_list, self.metrics)

    def prune_model(self, weights, rate):
        pruned_weights = []
        for layer in weights:
            if len(layer.shape) > 1:
                flat_weights = layer.flatten()
                threshold = np.percentile(np.abs(flat_weights), rate * 100)
                pruned_layer = np.where(np.abs(layer) < threshold, 0, layer)
                pruned_weights.append(pruned_layer)
            else:
                pruned_weights.append(layer)
        return pruned_weights

    def agg_weights(self):
        if not self.client_training_response:
            print("No client responses to aggregate. Keeping global model as is.")
            return self.global_model_weights
        
        recovered_models = []
        for client_id, response in self.client_training_response.items():
            recovered_models.append((response['weights'], response['num_samples']))
            reward = self.calculate_reward(response['metrics'])
            # Ensure client_pruning_rates has the key before updating bandit
            if client_id in self.client_pruning_rates:
                self.bandit.update(self.client_pruning_rates[client_id], reward, self.metrics.get(client_id, {}))
        
        total_samples = sum(s for _, s in recovered_models)
        if total_samples == 0:
            return self.global_model_weights

        agg_weights = [np.zeros_like(layer) for layer in self.global_model_weights]
        for model_weights, num_samples in recovered_models:
            scaling_factor = num_samples / total_samples
            model_weights_np = [np.asarray(w, dtype=np.float32) for w in model_weights]
            for i, layer in enumerate(model_weights_np):
                agg_weights[i] += layer * scaling_factor
        
        self.global_model_weights = agg_weights
        self.client_training_response.clear()
        return self.global_model_weights

    # --- THIS IS THE CORRECTED FUNCTION ---
    def calculate_reward(self, client_metrics):
        accuracy = client_metrics.get('accuracy', 0)
        energy = client_metrics.get('energy_consumption') # Get the value
        
        # If energy is None (because the file wasn't found), default to 1.0
        if energy is None:
            energy = 1.0
            
        reward = accuracy / (energy + 1e-6)
        return reward