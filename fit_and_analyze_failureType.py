import os
import json
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, mannwhitneyu

# --- Configuration ---
LOG_DIRECTORY = "experiment_logs"
R_ALERT_THRESHOLD = 0.5
TIME_PER_STEP_SECONDS = 5
BASELINE_THRESHOLDS = {'loss_spike_std': 3.0, 'grad_norm_abs': 50.0, 'lambda_abs': 40.0}

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
            'fault_type': config['fault_injection']['type'] # *** ADDED FAULT TYPE FOR BREAKDOWN ***
        }
        
        for line in lines:
            if line.get('event') == 'METRICS':
                r_value = np.exp(-sum([
                    get_weibull_term(line['lambda'], fitted_params['lambda']['eta'], fitted_params['lambda']['beta']),
                    get_weibull_term(line['sigma_sq'], fitted_params['sigma_sq']['eta'], fitted_params['sigma_sq']['beta']),
                    get_weibull_term(line['delta_l'], fitted_params['delta_l']['eta'], fitted_params['delta_l']['beta'])
                ]))
                if r_value < R_ALERT_THRESHOLD and summary['r_alert_step'] is None:
                    summary['r_alert_step'] = line['step']
                    break
        run_summaries.append(summary)
    return pd.DataFrame(run_summaries)

def print_performance_report(df, method_name, alert_col_name):
    """Calculates and prints a full performance report for a given method."""
    tp = len(df[(df['is_failure'] == True) & (df[alert_col_name].notna())])
    fp = len(df[(df['is_failure'] == False) & (df[alert_col_name].notna())])
    fn = len(df[(df['is_failure'] == True) & (df[alert_col_name].isna())])
    failure_count = tp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"| {method_name:<28} | {failure_count:^14} | {precision:^11.3f} | {recall:^8.3f} | {f1_score:^10.3f} |")

def main():
    # Action 1: Parse logs and fit parameters
    print("--- Step 1: Parsing logs and fitting Weibull parameters ---")
    all_signal_data = parse_all_logs_for_fitting(LOG_DIRECTORY)
    fitted_params = {}
    for signal_name, data in all_signal_data.items():
        eta, beta = fit_weibull_parameters(data)
        fitted_params[signal_name] = {'eta': eta, 'beta': beta}
        print(f"Fitted params for '{signal_name}':\teta (scale) = {eta:.4f}, beta (shape) = {beta:.4f}")

    # Action 2: Analyze all runs
    results_df = analyze_all_runs(LOG_DIRECTORY, fitted_params)
    
    # --- Action 3: Generate and print the performance breakdown for Table 3 ---
    print("\n" + "="*80)
    print("--- Performance by Failure Type (for Table 3) ---")
    print(f"| {'Failure Type':<28} | {'Failure Count':^14} | {'Precision':^11} | {'Recall':^8} | {'F1-Score':^10} |")
    print(f"|{'-'*30}|{'-'*16}|{'-'*13}|{'-'*10}|{'-'*12}|")
    
    # Map simulator fault types to descriptive names for the table
    fault_type_map = {
        'NODE_FAILURE': 'Hardware Cascade',
        'GRADIENT_EXPLOSION': 'Gradient Explosion',
        'LR_SPIKE': 'Loss Divergence (LR)',
        'DATA_CORRUPTION': 'Loss Divergence (Data)'
    }
    
    # Calculate overall metrics for all fault types combined
    fault_runs_df = results_df[results_df['fault_type'] != 'NONE']
    print_performance_report(fault_runs_df, 'Overall', 'r_alert_step')
    print(f"|{'-'*30}|{'-'*16}|{'-'*13}|{'-'*10}|{'-'*12}|")

    # Calculate metrics for each specific fault type
    for fault_code, descriptive_name in fault_type_map.items():
        df_subset = results_df[results_df['fault_type'] == fault_code]
        if not df_subset.empty:
            print_performance_report(df_subset, descriptive_name, 'r_alert_step')
    
    print("="*80)

if __name__ == "__main__":
    main()
