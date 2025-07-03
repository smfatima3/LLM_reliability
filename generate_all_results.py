import os
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# --- Configuration ---
# *** CRITICAL: Set this to the parent directory where your 'logs_*' folders are located. ***
BASE_LOG_PATH = "/content/" 

TIME_PER_STEP_SECONDS = 5
FINAL_METRIC_THRESHOLD = 0.57

def parse_log_directory(log_directory):
    """Parses all log files in a single directory into a list of structured run data."""
    all_runs_data = []
    # Use the BASE_LOG_PATH to construct the full path
    full_path = os.path.join(BASE_LOG_PATH, log_directory)
    
    if not os.path.exists(full_path):
        print(f"Warning: Directory not found, skipping: {full_path}")
        return all_runs_data

    for filename in os.listdir(full_path):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(full_path, filename)
        with open(filepath, 'r') as f:
            try:
                lines = [json.loads(line) for line in f]
                if not lines: continue
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in file, skipping: {filename}")
                continue
        
        config = lines[0].get('config', {})
        final_event = lines[-1]
        is_failure = final_event.get('event') == 'EXPERIMENT_FAILURE'
        
        metrics_history = [line for line in lines if line.get('event') == 'METRICS']
        
        all_runs_data.append({
            'is_failure': is_failure,
            'fault_type': config.get('fault_injection', {}).get('type', 'UNKNOWN'),
            'metrics_history': metrics_history
        })
    return all_runs_data

def analyze_performance(all_runs_data, alert_threshold):
    """Analyzes performance for a given set of runs and a specific alert threshold."""
    run_summaries = []
    if not all_runs_data:
        return pd.DataFrame()

    for run in all_runs_data:
        alert_step = None
        for metrics in run['metrics_history']:
            if metrics.get('r_metric', 0) > alert_threshold and alert_step is None:
                alert_step = metrics.get('step')
                # Don't break here, we need the last step for failure time
        
        last_step = run['metrics_history'][-1].get('step') if run['metrics_history'] else 0
        
        run_summaries.append({
            'fault_type': run['fault_type'],
            'is_failure': run['is_failure'],
            'failure_step': last_step if run['is_failure'] else None,
            'alert_step': alert_step
        })
    
    return pd.DataFrame(run_summaries)

def print_performance_report(df, method_name):
    """Calculates and prints a full performance report for a given method."""
    if df.empty:
        print(f"| {method_name:<20} | {'N/A':^9} | {'N/A':^8} | {'N/A':^10} | {'N/A':^7} |")
        return

    tp = len(df[(df['is_failure'] == True) & (df['alert_step'].notna())])
    fp = len(df[(df['is_failure'] == False) & (df['alert_step'].notna())])
    fn = len(df[(df['is_failure'] == True) & (df['alert_step'].isna())])
    tn = len(df[(df['is_failure'] == False) & (df['alert_step'].isna())])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    print(f"| {method_name:<20} | {precision:^9.3f} | {recall:^8.3f} | {f1:^10.3f} | {fpr:^7.3f} |")

def main():
    # --- Load Data for All Ablation Modes ---
    print("--- Parsing all log files from all ablation runs ---")
    data_full = parse_log_directory('logs_FULL_METRIC')
    data_lambda = parse_log_directory('logs_LAMBDA_ONLY')
    data_sigma = parse_log_directory('logs_SIGMA_ONLY')
    data_delta_l = parse_log_directory('logs_DELTA_L_ONLY')
    
    # --- Analyze and Print the Ablation Study Results ---
    print("\n" + "="*70)
    print("--- Ablation Study Results ---")
    print("="*70)
    print(f"| {'Method':<20} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'FPR':^7} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*9}|")

    results_full = analyze_performance(data_full, FINAL_METRIC_THRESHOLD)
    print_performance_report(results_full, 'FULL_METRIC')

    results_lambda = analyze_performance(data_lambda, FINAL_METRIC_THRESHOLD)
    print_performance_report(results_lambda, 'LAMBDA_ONLY')

    results_sigma = analyze_performance(data_sigma, FINAL_METRIC_THRESHOLD)
    print_performance_report(results_sigma, 'SIGMA_ONLY')

    results_delta_l = analyze_performance(data_delta_l, FINAL_METRIC_THRESHOLD)
    print_performance_report(results_delta_l, 'DELTA_L_ONLY')
    print("="*70)
    
    # --- Detailed Analysis of the Winning Metric ---
    print("\n" + "="*70)
    print("--- Detailed Analysis of Final Metric (DELTA_L_ONLY) ---")
    print("="*70)
    
    final_df = results_delta_l
    
    if final_df.empty:
        print("No data found for DELTA_L_ONLY. Cannot perform detailed analysis.")
        return
        
    # Statistical Summary (Lead Time)
    detected_failures = final_df[(final_df['is_failure'] == True) & (final_df['alert_step'].notna())].copy()
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['alert_step']
        detected_failures['lead_time_minutes'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0
        
        print("\nStatistical Summary (Lead Time):")
        print(f"  - Mean Lead Time:         {detected_failures['lead_time_minutes'].mean():.2f} minutes")
        print(f"  - Median Lead Time:       {detected_failures['lead_time_minutes'].median():.2f} minutes")
    else:
        print("\nStatistical Summary (Lead Time): No failures detected to analyze.")

if __name__ == "__main__":
    main()
