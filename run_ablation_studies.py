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
        self.loss_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=1000)
        self.sigma_sq_history = deque(maxlen=1000)
        self.delta_l_history = deque(maxlen=1000)

    def _normalize(self, value, history):
        if not history or len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        return (value - min_val) / (max_val - min_val)

    def calculate_lambda(self):
        # This is a placeholder as lambda was found to be ineffective
        return 0.0

    def calculate_sigma_sq(self, all_worker_grad_norms):
        if not all_worker_grad_norms or len(all_worker_grad_norms) < 2: return 0.0
        val = np.var(all_worker_grad_norms); self.sigma_sq_history.append(val); return val

    def calculate_delta_l(self, current_val_loss):
        if np.isnan(current_val_loss): return 100.0
        if not self.loss_history: self.loss_history.append(current_val_loss); return 0.0
        moving_avg = np.mean([l for l in self.loss_history if not np.isnan(l)])
        val = current_val_loss - moving_avg; self.delta_l_history.append(val); return val

    def calculate_r_metric(self, lambda_val, sigma_sq_val, delta_l_val):
        r_metric = 0.0
        if self.metric_mode == 'FULL_METRIC':
            sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
            delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)
            r_metric = 0.5 * sigma_sq_norm + 0.78 * delta_l_norm # Using your tuned weights
        elif self.metric_mode == 'LAMBDA_ONLY':
            r_metric = self._normalize(lambda_val, self.lambda_history)
        elif self.metric_mode == 'SIGMA_ONLY':
            r_metric = self._normalize(sigma_sq_val, self.sigma_sq_history)
        elif self.metric_mode == 'DELTA_L_ONLY':
            r_metric = self._normalize(delta_l_val, self.delta_l_history)
        return {'r_metric': r_metric}

class FaultInjector:
    def __init__(self, config):
        self.config = config['fault_injection']; self.triggered = False
    def inject(self, step, simulator_state):
        if self.triggered or step != self.config['trigger_step']: return simulator_state, None
        self.triggered = True; fault_type = self.config['type']; log_message = f"Injecting fault: {fault_type}"
        if fault_type in ['LR_SPIKE', 'DATA_CORRUPTION', 'GRADIENT_EXPLOSION']:
            simulator_state['instability_mode'] = 'MODEL_INSTABILITY'
        return simulator_state, log_message

class TrainingSimulator:
    def __init__(self, config):
        self.config = config; self.monitor = ReliabilityMonitor(config); self.injector = FaultInjector(config); self.logger = self._setup_logger()
        self.state = {'step': 0, 'training_loss': 5.0, 'instability_mode': 'NONE', 'degradation_counter': 0, 'is_crashed': False}
        self.alerts = {}
        
    def _setup_logger(self):
        # *** THIS IS THE CORRECTED LOGIC ***
        # It creates a specific subdirectory for each ablation mode.
        log_dir = f"logs_{self.config.get('METRIC_MODE', 'default')}"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl")
        return open(log_path, 'w')

    def _log(self, data): self.logger.write(json.dumps(data) + '\n')
    
    def _get_simulated_grad_norms(self):
        base_norm = 10.0 * math.exp(-self.state['step'] / 2000) + 1.0
        norms = [random.uniform(base_norm * 0.9, base_norm * 1.1) for _ in range(self.config['num_workers'])]
        if self.state['instability_mode'] == 'MODEL_INSTABILITY':
            for i in range(len(norms)): norms[i] *= random.uniform(1, 20)
        return norms

    def _update_training_state(self):
        base_loss_decay = math.exp(-self.state['step'] / 3000)
        self.state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)
        if self.state['instability_mode'] != 'NONE':
            self.state['degradation_counter'] += 1
            self.state['training_loss'] += self.state['degradation_counter'] * 0.3 + random.uniform(0, 2.0)
            failure_probability = (self.state['degradation_counter'] / 15.0) ** 2
            if random.random() < failure_probability: self.state['is_crashed'] = True
            
    def run(self):
        self._log({'event': 'EXPERIMENT_START', 'config': self.config})
        for step in range(1, self.config['total_training_steps'] + 1):
            self.state['step'] = step
            self.state, fault_log = self.injector.inject(step, self.state)
            if fault_log: self._log({'step': step, 'event': 'FAULT_INJECTED', 'details': fault_log})
            self._update_training_state()
            if step % self.config['eval_every_n_steps'] == 0:
                self.state['validation_loss'] = self.state['training_loss'] + random.uniform(0.05, 0.1)
                grad_norms = self._get_simulated_grad_norms()
                r_metric_data = self.monitor.calculate_r_metric(0, self.monitor.calculate_sigma_sq(grad_norms), self.monitor.calculate_delta_l(self.state['validation_loss']))
                self._check_alerts(r_metric_data['r_metric'])
                log_data = {'step': step, 'event': 'METRICS', **r_metric_data, 'alerts': self.alerts}
                self._log(log_data)
            if self.state['is_crashed']:
                self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': "System Failure"}); break
        if not self.state['is_crashed']: self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
        self.logger.close()
        
    def _check_alerts(self, r_metric):
        if 'r_metric' not in self.alerts and r_metric > self.config['alert_thresholds']['r_metric']:
            self.alerts['r_metric'] = self.state['step']

if __name__ == '__main__':
    simulator = TrainingSimulator(CONFIG); simulator.run()
