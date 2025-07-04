import time
import random
import math
import numpy as np
import json
import os
from collections import deque

CONFIG = {
    'experiment_id': 'final_paper_run',
    'model_type': 'GPT-2',
    'METRIC_MODE': 'FULL_METRIC',
    'num_workers': 8,
    'total_training_steps': 5000,
    'eval_every_n_steps': 100,
    'fault_injection': { 'type': 'NODE_FAILURE', 'trigger_step': 2500, 'params': {} },
    'alert_thresholds': { 'r_metric': 0.57 }
}

class ReliabilityMonitor:
    def __init__(self, config):
        self.metric_mode = config.get('METRIC_MODE', 'FULL_METRIC')
        self.weights = config.get('r_metric_weights', {'w_sigma': 0.5, 'w_delta': 0.78})
        self.loss_history = deque(maxlen=100)
        self.sigma_sq_history = deque(maxlen=1000)
        self.delta_l_history = deque(maxlen=1000)

    def _normalize(self, value, history):
        if not history or len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        return (value - min_val) / (max_val - min_val)

    def calculate_sigma_sq(self, model_state):
        base_norm = 10.0 * math.exp(-model_state['step'] / 2000) + 1.0
        norms = [random.uniform(base_norm * 0.9, base_norm * 1.1) for _ in range(model_state['num_workers'])]
        if model_state['instability_mode'] == 'MODEL_INSTABILITY':
            for i in range(len(norms)): norms[i] *= random.uniform(1, 20)
        val = np.var(norms); self.sigma_sq_history.append(val); return val

    def calculate_delta_l(self, current_val_loss):
        if np.isnan(current_val_loss): return 100.0
        if not self.loss_history: self.loss_history.append(current_val_loss); return 0.0
        moving_avg = np.mean([l for l in self.loss_history if not np.isnan(l)])
        val = current_val_loss - moving_avg; self.delta_l_history.append(val); return val

    def get_instability_score(self, sigma_sq_val, delta_l_val):
        score = 0.0
        sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)
        
        if self.metric_mode == 'FULL_METRIC':
            score = self.weights['w_sigma'] * sigma_sq_norm + self.weights['w_delta'] * delta_l_norm
        elif self.metric_mode == 'SIGMA_ONLY':
            score = sigma_sq_norm
        elif self.metric_mode == 'DELTA_L_ONLY':
            score = delta_l_norm
        
        return score

class TrainingSimulator:
    def __init__(self, config):
        self.config = config
        self.monitor = ReliabilityMonitor(config)
        self.logger = self._setup_logger() # This must be called to create the logger

    def _setup_logger(self):
        # *** THIS IS THE CRUCIAL FUNCTION ***
        # It creates a specific subdirectory for each ablation mode.
        log_dir = f"logs_{self.config.get('METRIC_MODE', 'default')}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl")
        return open(log_path, 'w')
        
    def run(self):
        state = {
            'step': 0, 'training_loss': 5.0, 'instability_mode': 'NONE',
            'degradation_counter': 0, 'is_crashed': False, 'num_workers': 8
        }
        fault_triggered = False
        self.logger.write(json.dumps({'event': 'EXPERIMENT_START', 'config': self.config}) + '\n')

        for step in range(1, self.config['total_training_steps'] + 1):
            state['step'] = step
            # Fault Injection
            if not fault_triggered and step == self.config['fault_injection']['trigger_step']:
                fault_triggered = True
                if self.config['fault_injection']['type'] in ['LR_SPIKE', 'DATA_CORRUPTION', 'GRADIENT_EXPLOSION']:
                    state['instability_mode'] = 'MODEL_INSTABILITY'

            # Update State
            base_loss_decay = math.exp(-step / 3000)
            state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)
            if state['instability_mode'] != 'NONE':
                state['degradation_counter'] += 1
                state['training_loss'] += state['degradation_counter'] * 0.3 + random.uniform(0, 2.0)
                failure_probability = (state['degradation_counter'] / 15.0) ** 2
                if random.random() < failure_probability: state['is_crashed'] = True

            # Logging
            if step % self.config['eval_every_n_steps'] == 0:
                validation_loss = state['training_loss'] + random.uniform(0.05, 0.1)
                sigma_sq = self.monitor.calculate_sigma_sq(state)
                delta_l = self.monitor.calculate_delta_l(validation_loss)
                instability_score = self.monitor.get_instability_score(sigma_sq, delta_l)
                self.logger.write(json.dumps({'step': step, 'event': 'METRICS', 'r_metric': instability_score}) + '\n')
            
            if state['is_crashed']:
                self.logger.write(json.dumps({'step': step, 'event': 'EXPERIMENT_FAILURE'}) + '\n')
                break
        
        if not state['is_crashed']:
            self.logger.write(json.dumps({'step': step, 'event': 'EXPERIMENT_SUCCESS'}) + '\n')
        self.logger.close()

if __name__ == '__main__':
    simulator = TrainingSimulator(CONFIG); simulator.run()
