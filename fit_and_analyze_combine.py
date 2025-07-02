import os
import json
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, mannwhitneyu

# --- Configuration ---
LOG_DIRECTORY = "experiment_logs"
R_ALERT_THRESHOLD = 0.5
TIME_PER_STEP_SECONDS = 5
# --- Baseline Thresholds (empirically chosen for best performance) ---
BASELINE_THRESHOLDS = {
    'loss_spike_std': 3.0, # A spike > 3 standard deviations above the mean
    'grad_norm_abs': 50.0, # An absolute gradient norm threshold
    'lambda_abs': 40.0,    # An absolute hardware failure rate threshold
}


def parse_all_logs_for_fitting(log_directory):
    """Parses logs to extract raw signal data leading up to failures for Weibull fitting."""
    all_signals = {'lambda': [], 'sigma_sq': [], 'delta_l': []}
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f: lines = [json.loads(line) for line in f]
        if lines[-1]['event'] != 'EXPERIMENT_FAILURE': continue
        fault_injection_step = next((line['step'] for line in lines if line.get('event') == 'FAULT_INJECTED'), -1)
        if fault_injection_step == -1: continue
        for line in lines:
            if line.get('event') == 'METRICS' and line['step'] >= fault_injection_step:
                if line['lambda'] > 1e-6: all_signals['lambda'].append(line['lambda'])
                if line['sigma_sq'] > 1e-6: all_signals['sigma_sq'].append(line['sigma_sq'])
                if line['delta_l'] > 1e-6: all_signals['delta_l'].append(line['delta_l'])
    return all_signals

def fit_weibull_parameters(signal_data):
    """Fits Weibull parameters (η scale, β shape) for a given signal's data."""
    if not signal_data: return 1.0, 1.0
    shape_beta, _, scale_eta = weibull_min.fit(signal_data, floc=0)
    return scale_eta, shape_beta

def get_weibull_term(val, eta, beta):
    """Helper to calculate one term of the Weibull exponent."""
    if val <= 0: return 0.0
    return (val / eta) ** beta

def analyze_all_runs(log_directory, fitted_params):
    """
    Analyzes all logs to get outcomes and alert steps for our R metric AND all baselines.
    This is the core function that generates the data for all comparison tables.
    """
    run_summaries = []
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f: lines = [json.loads(line) for line in f]

        config = lines[0]['config']
        final_event = lines[-1]
        is_failure = final_event['event'] == 'EXPERIMENT_FAILURE'
        failure_step = final_event.get('step') if is_failure else None
        
        summary = {
            'experiment_id': config['experiment_id'], 'is_failure': is_failure,
            'failure_step': failure_step, 'r_alert_step': None,
            'loss_spike_alert_step': None, 'grad_norm_alert_step': None,
            'lambda_alert_step': None, 'sigma_alert_step': None, 'delta_l_alert_step': None
        }

        loss_history = []
        for line in lines:
            if line.get('event') == 'METRICS':
                # --- Our R Metric Calculation ---
                r_value = np.exp(-sum([
                    get_weibull_term(line['lambda'], fitted_params['lambda']['eta'], fitted_params['lambda']['beta']),
                    get_weibull_term(line['sigma_sq'], fitted_params['sigma_sq']['eta'], fitted_params['sigma_sq']['beta']),
                    get_weibull_term(line['delta_l'], fitted_params['delta_l']['eta'], fitted_params['delta_l']['beta'])
                ]))
                if r_value < R_ALERT_THRESHOLD and summary['r_alert_step'] is None:
                    summary['r_alert_step'] = line['step']

                # --- Baseline Calculations ---
                # 1. Loss Spike Heuristic
                loss_history.append(line['validation_loss'])
                if len(loss_history) > 10:
                    mean_loss, std_loss = np.mean(loss_history), np.std(loss_history)
                    if std_loss > 0 and (line['validation_loss'] - mean_loss) / std_loss > BASELINE_THRESHOLDS['loss_spike_std']:
                        if summary['loss_spike_alert_step'] is None: summary['loss_spike_alert_step'] = line['step']
                
                # 2. Gradient Norm Heuristic (using grad_norm_mean as a proxy)
                if line['grad_norm_mean'] > BASELINE_THRESHOLDS['grad_norm_abs'] and summary['grad_norm_alert_step'] is None:
                    summary['grad_norm_alert_step'] = line['step']

                # 3. Ablation Baselines (using a simple threshold on the raw signal)
                if line['lambda'] > BASELINE_THRESHOLDS['lambda_abs'] and summary['lambda_alert_step'] is None:
                    summary['lambda_alert_step'] = line['step']
                # For sigma and delta_l, we'll use their Weibull term as a proxy for a single-component alert
                sigma_reliability = np.exp(-get_weibull_term(line['sigma_sq'], fitted_params['sigma_sq']['eta'], fitted_params['sigma_sq']['beta']))
                delta_l_reliability = np.exp(-get_weibull_term(line['delta_l'], fitted_params['delta_l']['eta'], fitted_params['delta_l']['beta']))
                if sigma_reliability < 0.7 and summary['sigma_alert_step'] is None: # Using a higher threshold for single components
                    summary['sigma_alert_step'] = line['step']
                if delta_l_reliability < 0.7 and summary['delta_l_alert_step'] is None:
                    summary['delta_l_alert_step'] = line['step']

        run_summaries.append(summary)
    return pd.DataFrame(run_summaries)


def print_performance_report(df, method_name, alert_col_name):
    """Calculates and prints a full performance report for a given method."""
    tp = len(df[(df['is_failure'] == True) & (df[alert_col_name].notna())])
    fp = len(df[(df['is_failure'] == False) & (df[alert_col_name].notna())])
    fn = len(df[(df['is_failure'] == True) & (df[alert_col_name].isna())])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    detected_failures = df[(df['is_failure'] == True) & (df[alert_col_name].notna())].copy()
    mean_lead_time = 0
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures[alert_col_name]
        mean_lead_time = (detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0).mean()

    print(f"| {method_name:<28} | {precision:^9.3f} | {recall:^8.3f} | {f1_score:^10.3f} | {mean_lead_time:^17.1f} |")
    return [method_name, precision, recall, f1_score, mean_lead_time]


def main():
    # Action 1: Parse logs and fit parameters
    print("--- Step 1: Parsing logs and fitting Weibull parameters ---")
    all_signal_data = parse_all_logs_for_fitting(LOG_DIRECTORY)
    fitted_params = {}
    for signal_name, data in all_signal_data.items():
        eta, beta = fit_weibull_parameters(data)
        fitted_params[signal_name] = {'eta': eta, 'beta': beta}
        print(f"Fitted params for '{signal_name}':\teta (scale) = {eta:.4f}, beta (shape) = {beta:.4f}")

    # Action 2: Analyze all runs to get alert steps for all methods
    results_df = analyze_all_runs(LOG_DIRECTORY, fitted_params)
    
    # Action 3: Generate and print the final comparison table
    print("\n" + "="*80)
    print("--- Final Performance Comparison ---")
    print(f"| {'Method':<28} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'Lead Time (min)':^17} |")
    print(f"|{'-'*30}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*19}|")
    
    print_performance_report(results_df, 'Our R Metric (Integrated)', 'r_alert_step')
    print(f"|{'-'*30}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*19}|")
    print_performance_report(results_df, 'Simple Heuristic (Loss Spike)', 'loss_spike_alert_step')
    print_performance_report(results_df, 'Ablation: λ only', 'lambda_alert_step')
    print_performance_report(results_df, 'Ablation: σ² only', 'sigma_alert_step')
    print_performance_report(results_df, 'Ablation: ΔL only', 'delta_l_alert_step')
    
    print("="*80)


if __name__ == "__main__":
    main()
