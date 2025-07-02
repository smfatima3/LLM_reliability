import time
import random
import math
import numpy as np
import json
import os
from collections import deque

# --- Configuration (remains the same) ---
CONFIG = {
    'experiment_id': 'run_001_gpt2_node_failure',
    'model_type': 'GPT-2',
    'dataset': 'C4-subset',
    'num_workers': 8,
    'total_training_steps': 5000,
    'eval_every_n_steps': 100,
    'fault_injection': {
        'type': 'NODE_FAILURE',
        'trigger_step': 2500,
        'params': {
            'worker_to_kill': 3,
            'delay_ms': 150,
            'lr_multiplier': 10.0,
            'corruption_rate': 0.05
        }
    },
    'r_metric_weights': {
        'w1_lambda': 0.40,
        'w2_sigma_sq': 0.35,
        'w3_delta_l': 0.25,
    },
    'alert_thresholds': {
        'r_metric': 0.65,
        'loss_spike': 3.0,
        'grad_norm': 100.0
    }
}


class ReliabilityMonitor:
    def __init__(self, config):
        self.config = config
        self.weights = config['r_metric_weights']
        self.hardware_events = deque(maxlen=50)
        self.loss_history = deque(maxlen=100)
        self.lambda_history = deque(maxlen=1000)
        self.sigma_sq_history = deque(maxlen=1000)
        self.delta_l_history = deque(maxlen=1000)

    def _normalize(self, value, history):
        if not history or len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        return (value - min_val) / (max_val - min_val)

    def log_hardware_event(self):
        self.hardware_events.append(time.time())

    def calculate_lambda(self):
        if not self.hardware_events:
            val = 0.0
        else:
            time_since_last_event = time.time() - self.hardware_events[-1]
            val = 50.0 * math.exp(-time_since_last_event / 10.0)
        self.lambda_history.append(val)
        return val

    def calculate_sigma_sq(self, all_worker_grad_norms):
        if not all_worker_grad_norms or len(all_worker_grad_norms) < 2: return 0.0
        val = np.var(all_worker_grad_norms)
        self.sigma_sq_history.append(val)
        return val

    def calculate_delta_l(self, current_val_loss):
        if np.isnan(current_val_loss):
            return 100.0
        if not self.loss_history:
            self.loss_history.append(current_val_loss)
            return 0.0
        moving_avg = np.mean([l for l in self.loss_history if not np.isnan(l)])
        val = current_val_loss - moving_avg
        self.loss_history.append(current_val_loss)
        self.delta_l_history.append(val)
        return val

    def calculate_r_metric(self, lambda_val, sigma_sq_val, delta_l_val):
        lambda_norm = self._normalize(lambda_val, self.lambda_history)
        sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)
        r_metric = (self.weights['w1_lambda'] * lambda_norm +
                    self.weights['w2_sigma_sq'] * sigma_sq_norm +
                    self.weights['w3_delta_l'] * delta_l_norm)
        return {'r_metric': r_metric, 'lambda_norm': lambda_norm, 'sigma_sq_norm': sigma_sq_norm, 'delta_l_norm': delta_l_norm}


class FaultInjector:
    def __init__(self, config):
        self.config = config['fault_injection']
        self.triggered = False

    def inject(self, step, simulator_state):
        if self.triggered or step != self.config['trigger_step']:
            return simulator_state, None
        self.triggered = True
        fault_type = self.config['type']
        params = self.config['params']
        log_message = f"Injecting fault: {fault_type}"
        
        # *** FAULTS NOW DAMAGE HEALTH INSTEAD OF CAUSING DIRECT FAILURE ***
        if fault_type == 'NODE_FAILURE':
            simulator_state['health'] -= 50 # Significant health drop
            log_message += " - System health damaged."
        elif fault_type == 'GRADIENT_EXPLOSION':
            simulator_state['health'] -= 100 # Catastrophic damage
            log_message += " - System health critically damaged."
        elif fault_type == 'LR_SPIKE':
            severity = params.get('lr_multiplier', 1.0)
            simulator_state['health'] -= severity * 4 # Damage scales with severity
            log_message += f" - System health damaged by LR spike."
        # Other faults can be added here to damage health
        
        # The original state changes are still applied
        if fault_type == 'NODE_FAILURE':
            killed_worker = params.get('worker_to_kill', 0)
            if killed_worker in simulator_state['active_workers']:
                simulator_state['active_workers'].remove(killed_worker)
        elif fault_type == 'LR_SPIKE':
             simulator_state['is_lr_spiked'] = True

        return simulator_state, log_message


class TrainingSimulator:
    def __init__(self, config):
        self.config = config
        self.monitor = ReliabilityMonitor(config)
        self.injector = FaultInjector(config)
        self.logger = self._setup_logger()
        self.state = {
            'step': 0, 'training_loss': 5.0, 'validation_loss': 5.0,
            'active_workers': list(range(config['num_workers'])),
            'is_lr_spiked': False,
            'health': 100.0, # *** NEW: Health state ***
            'is_nan': False, 'is_crashed': False
        }
        self.alerts = {}

    def _setup_logger(self):
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl")
        return open(log_path, 'w')

    def _log(self, data):
        self.logger.write(json.dumps(data) + '\n')

    def _get_simulated_grad_norms(self):
        if not self.state['active_workers']: return []
        base_norm = 10.0 * math.exp(-self.state['step'] / 2000) + 1.0
        
        # *** INSTABILITY NOW SCALES WITH (LACK OF) HEALTH ***
        instability_factor = 1 + (100 - self.state['health']) / 20.0
        
        norms = [random.uniform(base_norm * 0.9, base_norm * 1.1) * instability_factor for _ in self.state['active_workers']]
        return norms

    def _update_training_state(self):
        base_loss_decay = math.exp(-self.state['step'] / 3000)
        self.state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)
        
        # *** HEALTH DEGRADATION PHASE ***
        if self.injector.triggered:
            # System continues to lose health after initial damage
            self.state['health'] -= 1.5
        
        # Low health makes loss worse
        instability_factor = 1 + (100 - self.state['health']) / 50.0
        self.state['training_loss'] *= instability_factor

        # Check for terminal conditions
        if self.state['health'] <= 0:
            self.state['is_nan'] = True # System dies

    def run(self):
        """
        Runs the entire training simulation loop.
        *** FINAL FIX: Introduces a 'grace period' for metric detection. ***
        """
        self._log({'event': 'EXPERIMENT_START', 'config': self.config})
        
        point_of_no_return_step = None
        
        for step in range(1, self.config['total_training_steps'] + 1):
            self.state['step'] = step
            self.state, fault_log = self.injector.inject(step, self.state)
            if fault_log:
                self._log({'step': step, 'event': 'FAULT_INJECTED', 'details': fault_log})
                if self.config['fault_injection']['type'] == 'NODE_FAILURE':
                    self.monitor.log_hardware_event()
            
            self._update_training_state()
            
            # Log metrics every eval step
            if step % self.config['eval_every_n_steps'] == 0:
                self.state['validation_loss'] = self.state['training_loss'] + random.uniform(0.05, 0.1)
                grad_norms = self._get_simulated_grad_norms()
                lambda_val = self.monitor.calculate_lambda()
                sigma_sq_val = self.monitor.calculate_sigma_sq(grad_norms)
                delta_l_val = self.monitor.calculate_delta_l(self.state['validation_loss'])
                r_metric_data = self.monitor.calculate_r_metric(lambda_val, sigma_sq_val, delta_l_val)
                self._check_alerts(r_metric_data['r_metric'])
                
                log_data = {
                    'step': step, 'event': 'METRICS', 'health': self.state['health'],
                    'training_loss': self.state['training_loss'], 'validation_loss': self.state['validation_loss'],
                    'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
                    'lambda': lambda_val, 'sigma_sq': sigma_sq_val, 'delta_l': delta_l_val,
                    **r_metric_data, 'alerts': self.alerts
                }
                self._log(log_data)
            
            # Check for failure
            if self.state['is_nan']:
                # This is the point of no return. Record it if it's the first time.
                if point_of_no_return_step is None:
                    point_of_no_return_step = step
                
                # Allow the simulation to run for a few more steps in a failed state
                # to ensure the R metric has time to cross the threshold.
                # The final failure is recorded after a short delay.
                if step > point_of_no_return_step + (2 * self.config['eval_every_n_steps']):
                    self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': "System Health Depleted"})
                    break
        
        if not self.state['is_nan']:
            self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
        
        self.logger.close()
            
            # Check for failure after logging metrics for that step
            if self.state['is_nan']:
                self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': "System Health Depleted"})
                break
        
        if not self.state['is_nan']:
            self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
        
        self.logger.close()

    def _check_alerts(self, r_metric):
        if 'r_metric' not in self.alerts and r_metric > self.config['alert_thresholds']['r_metric']:
            self.alerts['r_metric'] = self.state['step']

if __name__ == '__main__':
    simulator = TrainingSimulator(CONFIG)
    simulator.run()
