import time
import random
import math
import numpy as np
import json
import os
from collections import deque

# --- Configuration ---
# This dictionary defines a single experimental run. You will create 847
# versions of this to run all your experiments.

CONFIG = {
    'experiment_id': 'run_001_gpt2_node_failure',
    'model_type': 'GPT-2', # 'GPT-2', 'T5', 'BERT'
    'dataset': 'C4-subset',
    'num_workers': 8, # Simulating a TPU v3-8 pod
    'total_training_steps': 5000,
    'eval_every_n_steps': 100,

    # --- Fault Injection ---
    'fault_injection': {
        'type': 'NODE_FAILURE', # 'NODE_FAILURE', 'NETWORK_DEGRADATION', 'GRADIENT_EXPLOSION', 'LR_SPIKE', 'DATA_CORRUPTION', 'NONE'
        'trigger_step': 2500, # When to inject the fault
        'params': {
            'worker_to_kill': 3, # For NODE_FAILURE
            'delay_ms': 150, # For NETWORK_DEGRADATION
            'lr_multiplier': 10.0, # For LR_SPIKE
            'corruption_rate': 0.05 # For DATA_CORRUPTION
        }
    },

    # --- R Metric Weights (from your paper) ---
    'r_metric_weights': {
        'w1_lambda': 0.40,
        'w2_sigma_sq': 0.35,
        'w3_delta_l': 0.25,
    },

    # --- Thresholds for baselines and R metric ---
    'alert_thresholds': {
        'r_metric': 0.65,
        'loss_spike': 3.0, # In standard deviations
        'grad_norm': 100.0
    }
}

# --- Core Classes ---

class ReliabilityMonitor:
    """
    Calculates the components of the R metric (λ, σ², ΔL) and the metric itself.
    This class encapsulates the logic from Section 4.1 of your paper.
    """
    def __init__(self, config):
        self.config = config
        self.weights = config['r_metric_weights']

        # State for λ (Hardware Failure Rate)
        self.hardware_events = deque(maxlen=50) # Store timestamps of hardware events

        # State for ΔL (Validation Loss Drift)
        self.loss_history = deque(maxlen=100) # For moving average

        # State for Normalization
        self.lambda_history = deque(maxlen=1000)
        self.sigma_sq_history = deque(maxlen=1000)
        self.delta_l_history = deque(maxlen=1000)

    def _normalize(self, value, history):
        """Applies min-max normalization over a sliding window."""
        if not history:
            return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def log_hardware_event(self):
        """Simulates logging a hardware event like an ECC error or network drop."""
        self.hardware_events.append(time.time())

    def calculate_lambda(self):
        """
        Calculates λ based on the frequency of recent hardware events.
        This simulates the MTBF calculation over a sliding window.
        """
        if len(self.hardware_events) < 2:
            return 0.0 # Not enough data for a rate

        # Time window is from the first to the last event in the deque
        time_span_seconds = self.hardware_events[-1] - self.hardware_events[0]
        if time_span_seconds == 0:
            return 0.0

        # Calculate failures per hour for a more intuitive scale
        failures_per_second = len(self.hardware_events) / time_span_seconds
        lambda_val = failures_per_second * 3600 # Failures per hour
        self.lambda_history.append(lambda_val)
        return lambda_val

    def calculate_sigma_sq(self, all_worker_grad_norms):
        """
        Calculates σ² (gradient variance) across all workers.
        Input is a list of gradient norms, one from each worker.
        """
        if not all_worker_grad_norms:
            return 0.0
        sigma_sq_val = np.var(all_worker_grad_norms)
        self.sigma_sq_history.append(sigma_sq_val)
        return sigma_sq_val

    def calculate_delta_l(self, current_val_loss):
        """
        Calculates ΔL (validation loss drift) against a moving average.
        """
        if not self.loss_history:
            self.loss_history.append(current_val_loss)
            return 0.0

        moving_avg = np.mean(self.loss_history)
        delta_l_val = current_val_loss - moving_avg
        self.loss_history.append(current_val_loss)
        self.delta_l_history.append(delta_l_val)
        return delta_l_val

    def calculate_r_metric(self, lambda_val, sigma_sq_val, delta_l_val):
        """
        Calculates the final R metric using normalized, weighted components.
        """
        lambda_norm = self._normalize(lambda_val, self.lambda_history)
        sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)

        r_metric = (self.weights['w1_lambda'] * lambda_norm +
                    self.weights['w2_sigma_sq'] * sigma_sq_norm +
                    self.weights['w3_delta_l'] * delta_l_norm)

        return {
            'r_metric': r_metric,
            'lambda_norm': lambda_norm,
            'sigma_sq_norm': sigma_sq_norm,
            'delta_l_norm': delta_l_norm
        }

class FaultInjector:
    """
    Applies different fault types to the training simulation at a specified step.
    This class encapsulates the logic from Section 4.2 of your paper.
    """
    def __init__(self, config):
        self.config = config['fault_injection']
        self.triggered = False

    def inject(self, step, simulator_state):
        """Checks if a fault should be triggered and applies it."""
        if self.triggered or step != self.config['trigger_step']:
            return simulator_state, None # No fault injected

        self.triggered = True
        fault_type = self.config['type']
        params = self.config['params']
        log_message = f"Injecting fault: {fault_type}"

        if fault_type == 'NODE_FAILURE':
            killed_worker = params['worker_to_kill']
            simulator_state['active_workers'].remove(killed_worker)
            log_message += f" - Worker {killed_worker} killed."

        elif fault_type == 'NETWORK_DEGRADATION':
            simulator_state['network_delay_ms'] = params['delay_ms']
            log_message += f" - Network delay set to {params['delay_ms']}ms."

        elif fault_type == 'GRADIENT_EXPLOSION':
            # This is simulated by having the model produce a huge grad norm
            simulator_state['force_grad_explosion'] = True
            log_message += " - Forcing gradient explosion."

        elif fault_type == 'LR_SPIKE':
            simulator_state['learning_rate'] *= params['lr_multiplier']
            log_message += f" - LR spiked to {simulator_state['learning_rate']:.6f}."

        elif fault_type == 'DATA_CORRUPTION':
            simulator_state['data_corruption_rate'] = params['corruption_rate']
            log_message += f" - Data corruption rate set to {params['corruption_rate']}."

        return simulator_state, log_message


class TrainingSimulator:
    """
    Main class to run a single training simulation experiment.
    """
    def __init__(self, config):
        self.config = config
        self.monitor = ReliabilityMonitor(config)
        self.injector = FaultInjector(config)
        self.logger = self._setup_logger()

        # --- Initial Training State ---
        self.state = {
            'step': 0,
            'training_loss': 5.0, # Initial high loss
            'validation_loss': 5.0,
            'learning_rate': 1e-4,
            'active_workers': list(range(config['num_workers'])),
            'network_delay_ms': 0,
            'force_grad_explosion': False,
            'data_corruption_rate': 0.0,
            'is_nan': False,
            'is_crashed': False
        }
        self.alerts = {} # To store when each alert was first triggered

    def _setup_logger(self):
        """Sets up a structured JSON logger for the experiment."""
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{self.config['experiment_id']}.jsonl")
        return open(log_path, 'w')

    def _log(self, data):
        """Writes a JSON line to the log file."""
        self.logger.write(json.dumps(data) + '\n')

    def _get_simulated_grad_norms(self):
        """
        Simulates collecting gradient norms from each worker.
        This is where the effects of faults become visible.
        """
        if not self.state['active_workers']:
            return []

        # Base grad norm decreases as training progresses
        base_norm = 10.0 * math.exp(-self.state['step'] / 2000) + 1.0
        
        # Add some natural noise
        norms = [random.uniform(base_norm * 0.9, base_norm * 1.1) for _ in self.state['active_workers']]

        # Simulate effects of instabilities
        if self.state['force_grad_explosion']:
            norms[random.randint(0, len(norms)-1)] *= 50 # One worker explodes
            self.state['force_grad_explosion'] = False # One-time event

        if self.state['data_corruption_rate'] > 0:
             # Corrupted data leads to higher variance
            for i in range(len(norms)):
                if random.random() < self.state['data_corruption_rate']:
                    norms[i] *= random.uniform(1.5, 3.0)

        return norms

    def _update_training_state(self):
        """
        Simulates a single training step, updating loss and other state.
        This is a placeholder for the actual `model.forward()` and `optimizer.step()`.
        """
        # --- Simulate Loss ---
        # Loss should generally decrease
        base_loss_decay = math.exp(-self.state['step'] / 3000)
        self.state['training_loss'] = 3.0 * base_loss_decay + random.uniform(0.1, 0.2)

        # LR spike causes loss to increase
        if self.config['fault_injection']['type'] == 'LR_SPIKE' and self.injector.triggered:
            self.state['training_loss'] *= 1.5

        # Check for terminal conditions
        if self.state['training_loss'] > 50 or np.isnan(self.state['training_loss']):
            self.state['is_nan'] = True
        
        if not self.state['active_workers']:
            self.state['is_crashed'] = True


    def run(self):
        """Runs the entire training simulation loop."""
        print(f"--- Starting Experiment: {self.config['experiment_id']} ---")
        self._log({'event': 'EXPERIMENT_START', 'config': self.config})

        for step in range(1, self.config['total_training_steps'] + 1):
            self.state['step'] = step
            
            # 1. Inject fault if it's time
            self.state, fault_log = self.injector.inject(step, self.state)
            if fault_log:
                self._log({'step': step, 'event': 'FAULT_INJECTED', 'details': fault_log})
                # Simulate a hardware event log when a node dies
                if self.config['fault_injection']['type'] == 'NODE_FAILURE':
                    self.monitor.log_hardware_event()

            # 2. Simulate a training step
            time.sleep(self.state['network_delay_ms'] / 1000.0) # Simulate network delay
            self._update_training_state()

            # 3. Collect metrics for R calculation
            grad_norms = self._get_simulated_grad_norms()
            
            # We only evaluate and get new metrics every N steps
            if step % self.config['eval_every_n_steps'] == 0:
                # Update validation loss (simulated)
                self.state['validation_loss'] = self.state['training_loss'] + random.uniform(0.05, 0.1)

                # Calculate R metric components
                lambda_val = self.monitor.calculate_lambda()
                sigma_sq_val = self.monitor.calculate_sigma_sq(grad_norms)
                delta_l_val = self.monitor.calculate_delta_l(self.state['validation_loss'])
                
                # Calculate the final R metric
                r_metric_data = self.monitor.calculate_r_metric(lambda_val, sigma_sq_val, delta_l_val)

                # 4. Check for alerts
                self._check_alerts(r_metric_data['r_metric'], np.mean(grad_norms))

                # 5. Log everything
                log_data = {
                    'step': step,
                    'event': 'METRICS',
                    'training_loss': self.state['training_loss'],
                    'validation_loss': self.state['validation_loss'],
                    'grad_norm_mean': np.mean(grad_norms) if grad_norms else 0,
                    'lambda': lambda_val,
                    'sigma_sq': sigma_sq_val,
                    'delta_l': delta_l_val,
                    **r_metric_data,
                    'alerts': self.alerts
                }
                self._log(log_data)

            # 6. Check for end of experiment
            if self.state['is_nan'] or self.state['is_crashed']:
                end_reason = "NaN Loss" if self.state['is_nan'] else "Worker Crash"
                self._log({'step': step, 'event': 'EXPERIMENT_FAILURE', 'reason': end_reason})
                print(f"--- Experiment Failed at step {step}: {end_reason} ---")
                break
        
        if not (self.state['is_nan'] or self.state['is_crashed']):
            self._log({'step': step, 'event': 'EXPERIMENT_SUCCESS'})
            print(f"--- Experiment Completed Successfully ---")

        self.logger.close()

    def _check_alerts(self, r_metric, grad_norm):
        """Checks metric values against thresholds and records the first alert time."""
        thresholds = self.config['alert_thresholds']
        
        if 'r_metric' not in self.alerts and r_metric > thresholds['r_metric']:
            self.alerts['r_metric'] = self.state['step']
        
        if 'grad_norm' not in self.alerts and grad_norm > thresholds['grad_norm']:
            self.alerts['grad_norm'] = self.state['step']
        
        # For loss spike, we need a history
        loss_hist = list(self.monitor.loss_history)
        if len(loss_hist) > 10:
            mean_loss = np.mean(loss_hist)
            std_loss = np.std(loss_hist)
            if std_loss > 0 and (self.state['validation_loss'] - mean_loss) / std_loss > thresholds['loss_spike']:
                 if 'loss_spike' not in self.alerts:
                    self.alerts['loss_spike'] = self.state['step']


if __name__ == '__main__':
    # This is how you would run a single experiment.
    # To complete the paper, you would programmatically generate 847 different
    # CONFIG dictionaries and run the simulator for each one.
    
    # Example 1: The default node failure
    simulator = TrainingSimulator(CONFIG)
    simulator.run()

    # Example 2: A different experiment - LR Spike with BERT
    bert_lr_spike_config = CONFIG.copy()
    bert_lr_spike_config['experiment_id'] = 'run_002_bert_lr_spike'
    bert_lr_spike_config['model_type'] = 'BERT'
    bert_lr_spike_config['fault_injection'] = {
        'type': 'LR_SPIKE',
        'trigger_step': 3000,
        'params': {'lr_multiplier': 15.0}
    }
    simulator_2 = TrainingSimulator(bert_lr_spike_config)
    simulator_2.run()
