import time
import random
import math
import numpy as np
import json
import os
from collections import deque

# --- Configuration ---
CONFIG = {
    'experiment_id': 'ablation_lambda_only',
    'model_type': 'GPT-2',
    'dataset': 'C4-subset',
    'num_workers': 8,
    'total_training_steps': 5000,
    'eval_every_n_steps': 100,
    'fault_injection': { 'type': 'NODE_FAILURE', 'trigger_step': 2500, 'params': {} },
    'alert_thresholds': { 'r_metric': 0.55 } # R is now just normalized λ, so we alert when it's high
}

class ReliabilityMonitor:
    def __init__(self, config):
        self.config = config; self.hardware_events = deque(maxlen=50); self.lambda_history = deque(maxlen=1000)
    def _normalize(self, value, history):
        if not history or len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        return (value - min_val) / (max_val - min_val)
    def log_hardware_event(self): self.hardware_events.append(time.time())
    def calculate_lambda(self):
        if not self.hardware_events: val = 0.0
        else:
            # A recent hardware event causes a large, decaying spike in the signal
            time_since_last_event = time.time() - self.hardware_events[-1]
            val = 50.0 * math.exp(-time_since_last_event / 10.0)
        self.lambda_history.append(val); return val
    def calculate_sigma_sq(self, all_worker_grad_norms): return 0.0 # Not used
    def calculate_delta_l(self, current_val_loss): return 0.0 # Not used
    
    def calculate_r_metric(self, lambda_val, sigma_sq_val, delta_l_val):
        # --- KEY CHANGE: R is now based ONLY on the normalized Lambda ---
        # A high lambda is a sign of instability.
        lambda_norm = self._normalize(lambda_val, self.lambda_history)
        r_metric = lambda_norm
        return {'r_metric': r_metric, 'lambda_norm': lambda_norm, 'sigma_sq_norm': 0, 'delta_l_norm': 0}

class FaultInjector:
    def __init__(self, config):
        self.config = config['fault_injection']; self.triggered = False
    def inject(self, step, simulator_state):
        if self.triggered or step != self.config['trigger_step']: return simulator_state, None
        self.triggered = True; fault_type = self.config['type']; params = self.config['params']; log_message = f"Injecting fault: {fault_type}"
        # Even though we only monitor lambda, we still inject other faults
        # to see if lambda-only can detect them (it shouldn't, which is the point).
        if fault_type in ['LR_SPIKE', 'DATA_CORRUPTION']: simulator_state['instability_mode'] = 'LOSS_DIVERGENCE'
        elif fault_type == 'GRADIENT_EXPLOSION': simulator_state['instability_mode'] = 'GRADIENT_INSTABILITY'
        return simulator_state, log_message

class TrainingSimulator:
    def __init__(self, config):
        self.config = config; self.monitor = ReliabilityMonitor(config); self.injector = FaultInjector(config); self.logger = self._setup_logger()
        self.state = {'step': 0, 'training_loss': 5.0, 'active_workers': list(range(config['num_workers'])), 'instability_mode': 'NONE', 'degradation_counter': 0, 'is_crashed': False}
        self.alerts = {}
    def _setup_logger(self):
        log_dir = "experiment_logs"; os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl"); return open(log_path, 'w')
    def _log(self, data): self.logger.write(json.dumps(data) + '\n')
    def _get_simulated_grad_norms(self): return [] # Not used
    def _update_training_state(self):
        base_loss_decay = math.exp(-self.state['step'] / 3000)
        self.state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)
        if self.state['instability_mode'] != 'NONE':
            self.state['degradation_counter'] += 1
            if self.state['instability_mode'] == 'LOSS_DIVERGENCE':
                self.state['training_loss'] += self.state['degradation_counter'] * 0.2 + random.uniform(0, 1.5)
            failure_probability = (self.state['degradation_counter'] / 20.0) ** 2
            if random.random() < failure_probability: self.state['is_crashed'] = True
    def run(self):
        self._log({'event': 'EXPERIMENT_START', 'config': self.config})
        for step in range(1, self.config['total_training_steps'] + 1):
            self.state['step'] = step
            self.state, fault_log = self.injector.inject(step, self.state)
            if fault_log:
                self._log({'step': step, 'event': 'FAULT_INJECTED', 'details': fault_log})
                # This is the crucial part for this specific ablation
                if self.config['fault_injection']['type'] == 'NODE_FAILURE':
                    self.monitor.log_hardware_event()
            self._update_training_state()
            if step % self.config['eval_every_n_steps'] == 0:
                self.state['validation_loss'] = self.state['training_loss'] + random.uniform(0.05, 0.1)
                r_metric_data = self.monitor.calculate_r_metric(self.monitor.calculate_lambda(), 0, 0)
                self._check_alerts(r_metric_data['r_metric'])
                log_data = {'step': step, 'event': 'METRICS', 'training_loss': self.state['training_loss'], 'validation_loss': self.state['validation_loss'], 'grad_norm_mean': 0, 'lambda': self.monitor.lambda_history[-1], 'sigma_sq': 0, 'delta_l': 0, **r_metric_data, 'alerts': self.alerts}
                self._log(log_data)
            if self.state['is_crashed']:
                self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': "System Failure"}); break
        if not self.state['is_crashed']: self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
        self.logger.close()
    def _check_alerts(self, r_metric):
        # Alert if normalized lambda is HIGH
        if 'r_metric' not in self.alerts and r_metric > self.config['alert_thresholds']['r_metric']: self.alerts['r_metric'] = self.state['step']

if __name__ == '__main__':
    simulator = TrainingSimulator(CONFIG); simulator.run()
