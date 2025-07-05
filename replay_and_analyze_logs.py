import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from typing import Dict, Tuple

# --- Configuration ---
# TODO: Update this list with the names of your two historical failure log files.
LOG_FILES_TO_ANALYZE = [
    "/kaggle/working/LLM_reliability/run_002_gpt4_moe_expert_failure.jsonl", 
    "/kaggle/working/LLM_reliability/run_001_llama3_router_imbalance (2).jsonl"
]

# --- Corrected R-Metric Class (from our validated methodology) ---
class CorrectedRMetric:
    """
    An R-Metric implementation that uses the validated methodology from the paper,
    including min-max normalization and correct component weighting.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.lambda_history = deque(maxlen=window_size)
        self.sigma_history = deque(maxlen=window_size)
        self.delta_history = deque(maxlen=window_size)
        
        # Weights from the paper's experimental design
        self.weights = {
            'lambda': 0.40,
            'sigma_squared': 0.35,
            'delta_l': 0.25
        }
        self.base_threshold = 0.57
        
    def _normalize_component(self, value: float, history: deque) -> float:
        """Correct min-max normalization over a sliding window."""
        if len(history) < 2: return 0.0
        min_val, max_val = min(history), max(history)
        if max_val == min_val: return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def calculate_r_metric(self, lambda_hw: float, sigma_squared: float, delta_l: float) -> float:
        """Calculates the R-Metric using the corrected methodology."""
        self.lambda_history.append(lambda_hw)
        self.sigma_history.append(sigma_squared)
        self.delta_history.append(delta_l)
        
        lambda_norm = self._normalize_component(lambda_hw, self.lambda_history)
        sigma_norm = self._normalize_component(sigma_squared, self.sigma_history)
        delta_norm = self._normalize_component(delta_l, self.delta_history)
        
        r_metric = (
            self.weights['lambda'] * lambda_norm +
            self.weights['sigma_squared'] * sigma_norm +
            self.weights['delta_l'] * delta_norm
        )
        return r_metric
    
    def predict_failure(self, r_metric: float) -> bool:
        """Predicts failure based on the R-Metric and the validated threshold."""
        return r_metric > self.base_threshold

# --- Main Analysis and Plotting Logic ---
def replay_and_plot_log(log_file: str):
    """
    Reads a historical log file, replays it through the R-Metric calculator,
    and generates a detailed visualization.
    """
    print(f"--- Analyzing historical log: {log_file} ---")
    
    # 1. Parse the log file
    try:
        with open(log_file, 'r') as f:
            lines = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: Log file not found: {log_file}. Skipping.")
        return
        
    metrics_data = [line for line in lines if line.get('event') == 'METRICS']
    if not metrics_data:
        print(f"ERROR: No METRICS events found in {log_file}. Skipping.")
        return
        
    df = pd.DataFrame(metrics_data)
    
    # Extract key event steps
    fault_step = next((line.get('step') for line in lines if line.get('event') == 'FAULT_INJECTED'), None)
    failure_event = next((line for line in lines if line.get('event') == 'EXPERIMENT_FAILURE'), None)
    failure_step = failure_event.get('step') if failure_event else None

    # 2. Replay the log through our R-Metric logic
    r_metric_calculator = CorrectedRMetric()
    replay_results = []
    for index, row in df.iterrows():
        # Use .get() to handle potentially missing keys gracefully
        lambda_val = row.get('lambda', 0)
        sigma_val = row.get('sigma_sq', 0)
        delta_val = row.get('delta_l', 0)
        
        r_metric = r_metric_calculator.calculate_r_metric(lambda_val, sigma_val, delta_val)
        replay_results.append(r_metric)
        
    df['replayed_r_metric'] = replay_results
    
    # Find the first alert from our replayed metric
    alert_step = df[df['replayed_r_metric'] > r_metric_calculator.base_threshold]['step'].min()

    # 3. Generate the plot
    sns.set_theme(style="whitegrid", palette="colorblind")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Validation Loss
    sns.lineplot(x='step', y='validation_loss', data=df, ax=axes[0], label='Validation Loss', color='C0')
    axes[0].set_title(f'Case Study: {os.path.splitext(log_file)[0]}', fontsize=16, pad=20)
    axes[0].set_ylabel('Loss')
    
    # Plot 2: Key Raw Signals (Sigma and Delta)
    sns.lineplot(x='step', y='sigma_sq', data=df, ax=axes[1], label='Gradient Variance (σ²)', color='C1')
    ax2_twin = axes[1].twinx()
    sns.lineplot(x='step', y='delta_l', data=df, ax=ax2_twin, label='Loss Drift (ΔL)', color='C2')
    axes[1].set_ylabel('Gradient Variance', color='C1')
    ax2_twin.set_ylabel('Loss Drift', color='C2')
    
    # Plot 3: Replayed R-Metric
    sns.lineplot(x='step', y='replayed_r_metric', data=df, ax=axes[2], label='R-Metric', color='green')
    axes[2].axhline(y=r_metric_calculator.base_threshold, color='darkred', linestyle='--', lw=2, label=f'Alert Threshold ({r_metric_calculator.base_threshold})')
    axes[2].set_ylabel('R-Metric Score')
    axes[2].set_xlabel('Training Steps')

    # Add annotations for all key events
    for ax in axes:
        if fault_step:
            ax.axvline(x=fault_step, color='orange', linestyle='--', lw=2, label='Fault Injected')
        if failure_step:
            ax.axvline(x=failure_step, color='red', linestyle=':', lw=3, label='Terminal Failure')
        if not pd.isna(alert_step):
            ax.axvline(x=alert_step, color='purple', linestyle='-', lw=2, label='R-Metric Alert')

    # Create a single, consolidated legend
    handles, labels = [], []
    for ax in fig.axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_filename = f"analysis_{os.path.splitext(log_file)[0]}.png"
    plt.savefig(output_filename)
    print(f"SUCCESS: Generated analysis plot: {output_filename}\n")


if __name__ == "__main__":
    for log_file in LOG_FILES_TO_ANALYZE:
        replay_and_plot_log(log_file)
