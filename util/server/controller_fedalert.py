# util/server/controller_fedalert.py

import numpy as np
import copy
from scipy.stats import entropy

# Helper functions remain the same...
def flatten_updates(updates, old_model_weights):
    if not updates: return np.array([])
    diff_sum = [np.zeros_like(w) for w in old_model_weights]
    for client_weights in updates:
        for i, layer_weights in enumerate(client_weights):
            diff_sum[i] += (layer_weights - old_model_weights[i])
    avg_diff = [d / len(updates) for d in diff_sum]
    return np.concatenate([w.flatten() for w in avg_diff])

def calculate_kl_divergence(p_counts, q_counts, epsilon=1e-6):
    p_counts, q_counts = np.array(p_counts, dtype=float), np.array(q_counts, dtype=float)
    total_p, total_q = p_counts.sum(), q_counts.sum()
    if total_p == 0 or total_q == 0: return float('inf')
    alpha = epsilon * (total_p + total_q) / len(p_counts)
    p_smoothed, q_smoothed = p_counts + alpha, q_counts + alpha
    p, q = p_smoothed / p_smoothed.sum(), q_smoothed / q_smoothed.sum()
    return entropy(p, q)


class ControllerFedAlert:
    def __init__(self, server_args):
        self.args = server_args
        self.state = "STARTING"
        self.current_round = 0
        self.trainer_list = []
        self.metrics = {}
        self.client_responses = {}
        self.global_model_weights = None
        
        self.all_warmup_updates = []
        self.stage2_history = []
        self.baseline_update_dist = None
        self.baseline_bin_edges = None
        self.kl_threshold = float('inf')
        self.suspected_drifted_clients = set()
        self.model_checkpoints = []
        self.mitigation_active = False
        
        self.analysis_metrics = { 'drift_confirmations': [], 'accuracy': [], 'pre_mitigation_accuracy': [] }

    def next_action(self):
        if self.state == "STARTING":
            min_trainers = self.args.get("min_trainers", 1)
            if len(self.trainer_list) >= min_trainers and self.global_model_weights is not None:
                self.state = "WARMUP"
                self.current_round = 1
                return {"action": "log_phase", "phase": "WARM-UP"}, []
            return None, None 

        elif self.state == "WARMUP":
            if self.current_round > self.args.get("warmup_rounds", 10):
                self._establish_baseline()
                self.state = "MAIN_TRAINING"
                return {"action": "log_phase", "phase": "MAIN TRAINING, DETECTION & MITIGATION"}, []
            
            selected_clients = self._select_trainers_for_round()
            return {"action": "train_round"}, selected_clients

        elif self.state == "MAIN_TRAINING":
            if self.current_round > self.args.get("num_rounds", 50):
                self.state = "FINISHED"
                return None, None

            selected_clients = self._select_trainers_for_round()
            if self.args.get("drift_enabled") and self.current_round == self.args.get("drift_start_round"):
                num_drift_clients = int(len(self.trainer_list) * self.args.get("drift_percentage", 0.5))
                drift_indices = np.random.choice(len(self.trainer_list), num_drift_clients, replace=False)
                drift_client_ids = [self.trainer_list[i] for i in drift_indices]
                return {"action": "apply_drift", "drift_clients": drift_client_ids}, []
            
            return {"action": "train_round"}, selected_clients

        elif self.state == "FINISHED":
            return {"action": "stop"}, self.trainer_list

        return None, None

    # --- MODIFIED: process_responses now accepts a logger ---
    def process_responses(self, old_model_weights, logger):
        metricType = {"infotype": "METRIC"}
        execType = {"infotype": "EXECUT"}

        if not self.client_responses:
            logger.warning("No client responses received for this round. Skipping aggregation.", extra=execType)
            self.current_round += 1
            return None

        mitigation_action = None
        training_responses = {cid: resp for cid, resp in self.client_responses.items() if 'metrics' in resp}
        if not training_responses:
             logger.warning("No valid training responses received this round. Skipping logic.", extra=execType)
             self.current_round += 1
             return None
        
        logger.info(f"round: {self.current_round}", extra=metricType)

        if self.state == "WARMUP":
            updates = [resp['weights'] for resp in training_responses.values()]
            self.aggregate(updates)
            flat_update_batch = flatten_updates(updates, old_model_weights)
            self.all_warmup_updates.append(flat_update_batch)
            logger.info(f"Warm-up Round {self.current_round}/{self.args.get('warmup_rounds')} complete.", extra=execType)
        
        elif self.state == "MAIN_TRAINING":
            alerting_clients = {cid: resp for cid, resp in training_responses.items() if resp['metrics']['is_alert']}
            alert_rate = len(alerting_clients) / len(training_responses)
            is_drift_confirmed = False
            
            if alert_rate >= self.args.get("alert_trigger_threshold", 0.25):
                alerting_updates = [resp['weights'] for resp in alerting_clients.values()]
                is_drift = self._adaptive_stage_2_verification(alerting_updates, old_model_weights)
                self.stage2_history.append(is_drift)
                if len(self.stage2_history) > 2: self.stage2_history.pop(0)
                if len(self.stage2_history) == 2 and sum(self.stage2_history) >= 2:
                    is_drift_confirmed = True
            
            accuracies = [resp['metrics']['accuracy'] for resp in training_responses.values() if 'accuracy' in resp['metrics']]
            avg_accuracy = np.mean(accuracies) if accuracies else 0.0
            
            if is_drift_confirmed and self.args.get("mitigation_enabled", False):
                logger.info(f"DRIFT CONFIRMED at round {self.current_round}!", extra=execType)
                self.analysis_metrics['drift_confirmations'].append(self.current_round)
                if not self.mitigation_active:
                     self.analysis_metrics['pre_mitigation_accuracy'].append(avg_accuracy)
                mitigation_action = self._apply_mitigation_strategy(list(alerting_clients.keys()), {cid: resp['weights'] for cid, resp in training_responses.items()}, self.current_round)
                self.stage2_history = [False, False] 
            else:
                self.aggregate([resp['weights'] for resp in training_responses.values()])

            self.analysis_metrics['accuracy'].append(avg_accuracy)
            logger.info(f"mean_accuracy: {avg_accuracy}", extra=metricType)
            logger.info(f"alerts: {len(alerting_clients)}/{len(training_responses)}", extra=metricType)

        self.current_round += 1
        return mitigation_action

    def final_analysis(self, logger):
        metricType = {"infotype": "ANALYSIS"}
        logger.info("\n" + "="*50 + "\n--- FINAL SIMULATION ANALYSIS ---\n" + "="*50, extra=metricType)
        config = self.args
        if config.get("drift_enabled") and self.analysis_metrics['drift_confirmations']:
            delay = self.analysis_metrics['drift_confirmations'][0] - config.get("drift_start_round", 0)
            logger.info(f"✅ Detection Delay: {delay} rounds", extra=metricType)
        elif config.get("drift_enabled"):
            logger.info("❌ Drift was not confirmed.", extra=metricType)
        if config.get("mitigation_enabled") and self.mitigation_active:
            logger.info("\n📋 Mitigation Analysis:", extra=metricType)
            logger.info(f"  - Strategy: {config.get('mitigation_strategy')}", extra=metricType)
            if self.analysis_metrics['pre_mitigation_accuracy']:
                pre_acc = self.analysis_metrics['pre_mitigation_accuracy'][0]
                final_acc = self.analysis_metrics['accuracy'][-1]
                logger.info(f"  - Accuracy before mitigation: {pre_acc*100:.2f}%", extra=metricType)
                logger.info(f"  - Final accuracy: {final_acc*100:.2f}%", extra=metricType)
                logger.info(f"  - Accuracy Change: {(final_acc - pre_acc)*100:+.2f}%", extra=metricType)
        if self.analysis_metrics['accuracy']:
            logger.info(f"\n📉 Accuracy Analysis:\n  - Minimum accuracy reached: {min(self.analysis_metrics['accuracy'])*100:.2f}%", extra=metricType)

    # ... (other helper methods remain the same) ...
    def add_trainer(self, trainer_id): self.trainer_list.append(trainer_id)
    def update_metrics(self, trainer_id, metrics): self.metrics[trainer_id] = metrics
    def add_client_response(self, id, response): self.client_responses[id] = response
    def set_initial_global_model(self, weights):
        if self.global_model_weights is None: self.global_model_weights = [np.asarray(w) for w in weights]
    def _select_trainers_for_round(self):
        clients_per_round = self.args.get("clients_per_round", 1)
        return list(np.random.choice(self.trainer_list, clients_per_round, replace=False))
    def aggregate(self, client_updates):
        if not client_updates: return
        avg_weights = [np.zeros_like(w) for w in self.global_model_weights]
        for client_weights in client_updates:
            for i, layer_weights in enumerate(client_weights):
                avg_weights[i] += layer_weights
        avg_weights = [w / len(client_updates) for w in avg_weights]
        self.global_model_weights = avg_weights
    def _establish_baseline(self):
        all_updates = np.concatenate(self.all_warmup_updates)
        self.baseline_update_dist, self.baseline_bin_edges = np.histogram(all_updates, bins=50)
        bootstrap_kl = []
        for _ in range(300):
            s1 = np.random.choice(all_updates, size=len(all_updates)//4, replace=True)
            s2 = np.random.choice(all_updates, size=len(all_updates)//4, replace=True)
            d1, _ = np.histogram(s1, bins=self.baseline_bin_edges)
            d2, _ = np.histogram(s2, bins=self.baseline_bin_edges)
            kl = calculate_kl_divergence(d1, d2)
            if not np.isnan(kl) and np.isfinite(kl): bootstrap_kl.append(kl)
        if bootstrap_kl: self.kl_threshold = max(np.percentile(bootstrap_kl, 85), 0.02)
        else: self.kl_threshold = 0.05
    def _adaptive_stage_2_verification(self, alerting_updates, old_model_weights):
        flat_updates = flatten_updates(alerting_updates, old_model_weights)
        if flat_updates.size == 0: return False
        dist, _ = np.histogram(flat_updates, bins=self.baseline_bin_edges)
        kl = calculate_kl_divergence(self.baseline_update_dist, dist)
        print(f"    [Stage 2] KL Divergence: {kl:.6f} | Threshold: {self.kl_threshold:.6f}")
        return kl > self.kl_threshold
    def _apply_mitigation_strategy(self, alerting_ids, all_updates, round_num):
        strategy = self.args.get("mitigation_strategy", "hybrid")
        self.suspected_drifted_clients.update(alerting_ids)
        if not self.mitigation_active:
            self.mitigation_active = True
            print(f"    [Mitigation] Activated at round {round_num} (strategy: {strategy})")
        
        if strategy == "selective_aggregation":
            self._selective_aggregation(all_updates)
        elif strategy == "hybrid":
            self._selective_aggregation(all_updates)
            return {"action": "update_learning_rate", "client_ids": alerting_ids, "factor": self.args.get("adaptive_lr_factor")}
        return None
    def _selective_aggregation(self, all_updates):
        clean_updates = [up for cid, up in all_updates.items() if cid not in self.suspected_drifted_clients]
        if len(clean_updates) < self.args.get("min_clean_clients", 2):
            self.aggregate(list(all_updates.values()))
        else:
            self.aggregate(clean_updates)