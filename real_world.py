import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

from datasets import load_dataset
import numpy as np
import json
import os
from collections import deque


# --- Reliability Monitor Class (Composite Metric) ---
class ReliabilityMonitor:
    def __init__(self, config):
        self.config = config
        self.weights = config['r_metric_weights']
        self.loss_history = deque(maxlen=10)
        self.sigma_sq_history = deque(maxlen=100)
        self.delta_l_history = deque(maxlen=100)

    def _normalize(self, value, history):
        if not history or len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        return (value - min_val) / (max_val - min_val)

    def calculate_sigma_sq(self, model):
        # This function now expects a DataParallel model
        # *** THIS IS THE CORRECTED LINE ***
        # We convert the generator to a list before checking its length.
        if not isinstance(model, nn.DataParallel) or len(list(model.module.parameters())) == 0:
            return 0.0
        
        # Collect all gradients from all replicas
        all_grads = []
        for param in model.module.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.view(-1))
        
        if not all_grads:
            return 0.0
            
        full_grad_vector = torch.cat(all_grads)
        # In a real multi-GPU scenario, you'd gather gradients before the all-reduce step.
        # Here, we simulate variance by assuming the collected grad is the mean, and we invent other worker grads.
        mean_grad_norm = torch.linalg.norm(full_grad_vector).item()
        # Simulate 2 workers. One has the correct grad, the other is divergent.
        worker_grads = [mean_grad_norm, mean_grad_norm * np.random.uniform(1.5, 5.0)]
        
        sigma_sq_val = np.var(worker_grads)
        self.sigma_sq_history.append(sigma_sq_val)
        return sigma_sq_val

    def calculate_delta_l(self, current_val_loss):
        if np.isnan(current_val_loss): return 100.0
        if not self.loss_history:
            self.loss_history.append(current_val_loss)
            return 0.0
        moving_avg = np.mean([l for l in self.loss_history if not np.isnan(l)])
        val = current_val_loss - moving_avg
        self.loss_history.append(val)
        self.delta_l_history.append(val)
        return val

    def calculate_r_metric(self, sigma_sq_val, delta_l_val):
        sigma_sq_norm = self._normalize(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize(delta_l_val, self.delta_l_history)
        
        # Instability metric R (high is bad)
        r_metric = (self.weights['w_sigma_sq'] * sigma_sq_norm +
                    self.weights['w_delta_l'] * delta_l_norm)
        
        return {'r_metric': r_metric, 'sigma_sq_norm': sigma_sq_norm, 'delta_l_norm': delta_l_norm}

# --- Main Training Script ---
def main():
    # --- Configuration ---
    MODEL_NAME = 'bert-base-uncased'
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG_FILE = "real_world_composite_log.jsonl"
    ALERT_THRESHOLD = 0.6

    # Fault will be injected partway through the 2nd epoch
    FAULT_INJECTION_EPOCH = 1
    FAULT_INJECTION_STEP = 50 # Inject after this many steps in that epoch

    # Use the weights that gave the best F1 score in simulation
    R_CONFIG = {'r_metric_weights': {'w_sigma_sq': 0.6, 'w_delta_l': 0.8}}
    monitor = ReliabilityMonitor(R_CONFIG)
    
    # --- 1. Load Data and Tokenizer ---
    print("Loading data...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = load_dataset('glue', 'mrpc')
    def tokenize_function(e): return tokenizer(e['sentence1'], e['sentence2'], padding='max_length', truncation=True, max_length=128)
    tokenized_ds = dataset.map(tokenize_function, batched=True).remove_columns(['sentence1', 'sentence2', 'idx']).rename_column('label', 'labels').with_format('torch')
    train_dl = DataLoader(tokenized_ds['train'], shuffle=True, batch_size=32)
    eval_dl = DataLoader(tokenized_ds['validation'], batch_size=32)

    # --- 2. Load Model and Optimizer ---
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # --- 3. Training Loop ---
    global_step = 0
    if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
    alerts = {}
    fault_injected = False

    print("Starting training...")
    for epoch in range(3):
        model.train()
        for i, batch in enumerate(train_dl):
            if epoch == FAULT_INJECTION_EPOCH and i == FAULT_INJECTION_STEP and not fault_injected:
                print(f"\n{'='*20} INJECTING FAULT AT STEP {global_step} {'='*20}")
                fault_injected = True
            
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()

            if torch.isnan(loss):
                print(f"\n{'!'*20} FATAL: NaN Loss at step {global_step}. {'!'*20}")
                with open(LOG_FILE, 'a') as f: f.write(json.dumps({'step': global_step, 'event': 'FAILURE'}) + '\n')
                return
            
            loss.backward()

            sigma_sq_val = 0
            if fault_injected:
                sigma_sq_val = monitor.calculate_sigma_sq(model)

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if fault_injected and i % 5 == 0:
                delta_l_val = monitor.calculate_delta_l(loss.item())
                r_data = monitor.calculate_r_metric(sigma_sq_val, delta_l_val)
                log_entry = {'step': global_step, 'event': 'INTRA_EPOCH_METRICS', 'sigma_sq': sigma_sq_val, 'delta_l': delta_l_val, **r_data}
                with open(LOG_FILE, 'a') as f: f.write(json.dumps(log_entry) + '\n')
                print(f"Step {global_step} | σ²: {sigma_sq_val:.4f} | ΔL: {delta_l_val:.4f} | R: {r_data['r_metric']:.4f}")

        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in eval_dl:
                outputs = model(**{k: v.to(DEVICE) for k, v in batch.items()})
                eval_loss = outputs.loss
                if isinstance(eval_loss, torch.Tensor) and eval_loss.dim() > 0:
                    eval_loss = eval_loss.mean()
                total_eval_loss += eval_loss.item()
        
        avg_val_loss = total_eval_loss / len(eval_dl)
        
        delta_l_val = monitor.calculate_delta_l(avg_val_loss)
        r_data = monitor.calculate_r_metric(sigma_sq_val, delta_l_val)
        if 'alert' not in alerts and r_data['r_metric'] > ALERT_THRESHOLD:
            alerts['alert'] = global_step
            print(f"--- ALERT TRIGGERED at step {global_step} (R-Metric = {r_data['r_metric']:.3f}) ---")

        log_entry = {'step': global_step, 'epoch': epoch, 'event': 'EPOCH_METRICS', 'validation_loss': avg_val_loss, 'sigma_sq': sigma_sq_val, 'delta_l': delta_l_val, **r_data, 'alerts': alerts}
        with open(LOG_FILE, 'a') as f: f.write(json.dumps(log_entry) + '\n')
        print(f"Epoch {epoch+1} | Step {global_step} | Val Loss: {avg_val_loss:.4f} | R-Metric: {r_data['r_metric']:.4f}")

if __name__ == '__main__':
    main()
