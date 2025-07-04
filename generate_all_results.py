import os
import json
import pandas as pd
import numpy as np

# --- Configuration ---
LOG_DIRECTORY = "logs_FULL_METRIC" # We are analyzing our winning metric
FINAL_METRIC_THRESHOLD = 0.57
TIME_PER_STEP_SECONDS = 5
ACCELERATOR_COST_PER_HOUR = 2.50
NUM_ACCELERATORS = 8

def parse_log_directory(log_directory):
    """Parses all log files in a single directory."""
    all_runs_data = []
    if not os.path.exists(log_directory):
        print(f"Warning: Directory not found, skipping: {log_directory}")
        return all_runs_data
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]
        if not lines: continue
        is_failure = lines[-1].get('event') == 'EXPERIMENT_FAILURE'
        metrics_history = [line for line in lines if line.get('event') == 'METRICS']
        all_runs_data.append({'is_failure': is_failure, 'metrics_history': metrics_history})
    return all_runs_data

def analyze_performance(all_runs_data, alert_threshold, analysis_mode='R_METRIC'):
    """Analyzes performance for a given set of runs and a specific alert method."""
    run_summaries = []
    for run in all_runs_data:
        alert_step = None
        # --- Logic for different alert methods ---
        if analysis_mode == 'R_METRIC':
            for metrics in run['metrics_history']:
                if metrics.get('r_metric', 0) > alert_threshold and alert_step is None:
                    alert_step = metrics.get('step')
                    break
        elif analysis_mode == 'SLOPE_BASELINE':
            # Baseline: Alert if validation loss increases for 3 consecutive eval steps
            loss_values = [m.get('validation_loss', 1e6) for m in run['metrics_history']]
            for i in range(2, len(loss_values)):
                if loss_values[i] > loss_values[i-1] and loss_values[i-1] > loss_values[i-2]:
                    alert_step = run['metrics_history'][i].get('step')
                    break
        
        last_step = run['metrics_history'][-1].get('step') if run['metrics_history'] else 0
        run_summaries.append({
            'is_failure': run['is_failure'],
            'failure_step': last_step if run['is_failure'] else None,
            'alert_step': alert_step
        })
    
    df = pd.DataFrame(run_summaries)
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
    print("--- Parsing final experimental logs ---")
    final_metric_data = parse_log_directory(LOG_DIRECTORY)
    
    if not final_metric_data:
        print(f"Error: No log files found in '{LOG_DIRECTORY}'. Please run experiments first.")
        return

    # --- 1. Main Performance of Our R_ΔL Metric ---
    print("\n" + "="*70)
    print("--- Performance of Final R_ΔL Metric ---")
    print("="*70)
    core_metrics, results_df = analyze_performance(final_metric_data, FINAL_METRIC_THRESHOLD, 'R_METRIC')
    print(f"| {'Metric':<25} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'FPR':^7} |")
    print(f"|{'-'*27}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*9}|")
    print(f"| {'Our R_ΔL Metric':<25} | {core_metrics['precision']:^9.3f} | {core_metrics['recall']:^8.3f} | {core_metrics['f1']:^10.3f} | {core_metrics['fpr']:^7.3f} |")
    
    # --- 2. Comparison with Stronger Baseline ---
    print("\n" + "="*70)
    print("--- Comparison with Baselines ---")
    print("="*70)
    baseline_metrics, _ = analyze_performance(final_metric_data, 0, 'SLOPE_BASELINE') # Threshold not used for slope
    print(f"| {'Method':<25} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'FPR':^7} |")
    print(f"|{'-'*27}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*9}|")
    print(f"| {'Val. Loss Slope Baseline':<25} | {baseline_metrics['precision']:^9.3f} | {baseline_metrics['recall']:^8.3f} | {baseline_metrics['f1']:^10.3f} | {baseline_metrics['fpr']:^7.3f} |")
    print(f"| {'Our R_ΔL Metric':<25} | {core_metrics['precision']:^9.3f} | {core_metrics['recall']:^8.3f} | {core_metrics['f1']:^10.3f} | {core_metrics['fpr']:^7.3f} |")
    

    # --- 3. Threshold Sensitivity Analysis ---
    print("\n" + "="*70)
    print("--- Threshold Sensitivity Analysis (for Appendix) ---")
    print("="*70)
    print(f"| {'Alert Threshold (R > X)':<25} | {'F1-Score':<10} |")
    print(f"|{'-'*27}|{'-'*12}|")
    for threshold in np.arange(0.40, 0.75, 0.05):
        metrics, _ = analyze_performance(final_metric_data, threshold, 'R_METRIC')
        print(f"| {threshold:<25.2f} | {metrics['f1']:.3f}     |")
        
    # --- 4. Cost-Benefit Analysis ---
    print("\n" + "="*70)
    print("--- Cost-Benefit Analysis ---")
    print("="*70)
    detected_failures = results_df[(results_df['is_failure'] == True) & (results_df['alert_step'].notna())].copy()
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['alert_step']
        detected_failures['lead_time_hours'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 3600.0
        
        mean_lead_time_hours = detected_failures['lead_time_hours'].mean()
        total_runs_prevented = len(detected_failures)
        total_hours_saved = detected_failures['lead_time_hours'].sum() * NUM_ACCELERATORS
        total_cost_saved = total_hours_saved * ACCELERATOR_COST_PER_HOUR

        print(f"  - Average Lead Time:          {mean_lead_time_hours * 60:.2f} minutes")
        print(f"  - Total Failures Prevented:   {total_runs_prevented}")
        print(f"  - Total Accelerator-Hours Saved: {total_hours_saved:.2f} hours")
        print(f"  - Estimated Cost Savings:     ${total_cost_saved:,.2f}")
    else:
        print("  No detected failures to analyze for cost savings.")
    print("="*70)

if __name__ == "__main__":
    main()
