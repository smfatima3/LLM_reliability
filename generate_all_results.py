import os
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# --- Configuration ---
TIME_PER_STEP_SECONDS = 5
# This is the tuned threshold for our final, winning metric (DELTA_L_ONLY)
FINAL_METRIC_THRESHOLD = 0.57 

def parse_log_directory(log_directory):
    """Parses all log files in a single directory into a list of structured run data."""
    all_runs_data = []
    if not os.path.exists(log_directory):
        return all_runs_data

    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]
        
        config = lines[0]['config']
        final_event = lines[-1]
        is_failure = final_event['event'] == 'EXPERIMENT_FAILURE'
        
        metrics_history = [line for line in lines if line.get('event') == 'METRICS']
        
        all_runs_data.append({
            'is_failure': is_failure,
            'fault_type': config['fault_injection']['type'],
            'metrics_history': metrics_history
        })
    return all_runs_data

def analyze_performance(all_runs_data, alert_threshold):
    """Analyzes performance for a given set of runs and a specific alert threshold."""
    run_summaries = []
    for run in all_runs_data:
        alert_step = None
        for metrics in run['metrics_history']:
            # The 'r_metric' in the logs is the instability score for that specific run's mode
            if metrics.get('r_metric', 0) > alert_threshold and alert_step is None:
                alert_step = metrics['step']
                break
        
        last_step = run['metrics_history'][-1]['step'] if run['metrics_history'] else 0
        
        # --- THIS IS THE CORRECTED LOGIC ---
        # Ensure that 'is_failure' and 'fault_type' are correctly passed into the summary
        run_summaries.append({
            'fault_type': run['fault_type'],
            'is_failure': run['is_failure'],
            'failure_step': last_step if run['is_failure'] else None,
            'alert_step': alert_step
        })
    
    df = pd.DataFrame(run_summaries)
    
    # This part of the code will now work correctly because the 'is_failure' column exists
    tp = len(df[(df['is_failure'] == True) & (df['alert_step'].notna())])
    fp = len(df[(df['is_failure'] == False) & (df['alert_step'].notna())])
    fn = len(df[(df['is_failure'] == True) & (df['alert_step'].isna())])
    tn = len(df[(df['is_failure'] == False) & (df['alert_step'].isna())])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr}, df

def main():
    # --- Load Data for All Ablation Modes ---
    print("--- Parsing all log files from all ablation runs ---")
    data_full = parse_log_directory('logs_FULL_METRIC')
    data_lambda = parse_log_directory('logs_LAMBDA_ONLY')
    data_sigma = parse_log_directory('logs_SIGMA_ONLY')
    data_delta_l = parse_log_directory('logs_DELTA_L_ONLY')

    # --- Section 1: Core Performance of the Winning Metric (DELTA_L_ONLY) ---
    print("\n" + "="*60)
    print("--- Core Performance Results (Final Metric: DELTA_L_ONLY) ---")
    print("="*60)
    
    core_metrics, results_df = analyze_performance(data_delta_l, FINAL_METRIC_THRESHOLD)
    
    # 1. Statistical Summary (Lead Time)
    detected_failures = results_df[(results_df['is_failure'] == True) & (results_df['alert_step'].notna())].copy()
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['alert_step']
        detected_failures['lead_time_minutes'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0
        
        print("\n1. Statistical Summary (Lead Time):")
        print(f"  - Mean Lead Time:         {detected_failures['lead_time_minutes'].mean():.2f} minutes")
        print(f"  - Median Lead Time:       {detected_failures['lead_time_minutes'].median():.2f} minutes")
        print(f"  - 95th Percentile:        {detected_failures['lead_time_minutes'].quantile(0.95):.2f} minutes")
        
        mean_diff = detected_failures['lead_time_minutes'].mean() - 0.5
        pooled_std = detected_failures['lead_time_minutes'].std()
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        print(f"  - Effect Size (Cohen's d): {cohens_d:.2f}")
    else:
        print("\n1. Statistical Summary (Lead Time): No failures detected.")

    # 2. Cross-scenarios Analysis (Lead Time)
    print("\n2. Cross-scenarios Analysis (Lead Time by Fault Type):")
    if not detected_failures.empty:
        lead_time_by_type = detected_failures.groupby('fault_type')['lead_time_minutes'].mean()
        print(lead_time_by_type.round(2).to_string())
    else:
        print("  No detected failures to analyze.")

    # 3. Classification Result
    print("\n3. Classification Result:")
    print(f"  - Precision:              {core_metrics['precision']:.3f}")
    print(f"  - Recall:                 {core_docs['recall']:.3f}")
    print(f"  - F1-Score:               {core_metrics['f1']:.3f}")
    print(f"  - False Positive Rate:    {core_metrics['fpr']:.3f}")

    # --- Section 2: Component Contribution & Baseline Comparison ---
    print("\n" + "="*60)
    print("--- Component Contribution & Baselines (Ablation Study) ---")
    print("="*60)
    print(f"| {'Method':<20} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'FPR':^7} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*9}|")
    
    # Analyze and print for each mode
    metrics_full, _ = analyze_performance(data_full, FINAL_METRIC_THRESHOLD)
    print(f"| {'FULL_METRIC (Tuned)':<20} | {metrics_full['precision']:^9.3f} | {metrics_full['recall']:^8.3f} | {metrics_full['f1']:^10.3f} | {metrics_full['fpr']:^7.3f} |")
    
    metrics_lambda, _ = analyze_performance(data_lambda, FINAL_METRIC_THRESHOLD)
    print(f"| {'LAMBDA_ONLY':<20} | {metrics_lambda['precision']:^9.3f} | {metrics_lambda['recall']:^8.3f} | {metrics_lambda['f1']:^10.3f} | {metrics_lambda['fpr']:^7.3f} |")

    metrics_sigma, _ = analyze_performance(data_sigma, FINAL_METRIC_THRESHOLD)
    print(f"| {'SIGMA_ONLY':<20} | {metrics_sigma['precision']:^9.3f} | {metrics_sigma['recall']:^8.3f} | {metrics_sigma['f1']:^10.3f} | {metrics_sigma['fpr']:^7.3f} |")

    metrics_delta_l, _ = analyze_performance(data_delta_l, FINAL_METRIC_THRESHOLD)
    print(f"| {'DELTA_L_ONLY':<20} | {metrics_delta_l['precision']:^9.3f} | {metrics_delta_l['recall']:^8.3f} | {metrics_delta_l['f1']:^10.3f} | {metrics_delta_l['fpr']:^7.3f} |")


    # --- Section 3: Sensitivity & Robustness Analysis (for Table 4) ---
    print("\n" + "="*60)
    print("--- Sensitivity & Robustness (on DELTA_L_ONLY metric) ---")
    print("="*60)
    
    print("\nThreshold Robustness Analysis (F1-Score):")
    print(f"{'Threshold Multiplier':<25} | {'F1-Score':<10}")
    print("-" * 40)
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        threshold = FINAL_METRIC_THRESHOLD * multiplier
        metrics, _ = analyze_performance(data_delta_l, threshold)
        print(f"{multiplier:<25} | {metrics['f1']:.3f}")


if __name__ == "__main__":
    main()
