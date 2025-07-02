import os
import json
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, mannwhitneyu
from sklearn.metrics import roc_auc_score, roc_curve

# --- Configuration ---
LOG_DIRECTORY = "/kaggle/working/experiment_logs"
R_ALERT_THRESHOLD = 0.5
TIME_PER_STEP_SECONDS = 5

def parse_all_logs(log_directory):
    """Parses all log files into a list of structured run data."""
    all_runs_data = []
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]
        
        config = lines[0]['config']
        final_event = lines[-1]
        is_failure = final_event['event'] == 'EXPERIMENT_FAILURE'
        
        # Extract the full time-series of metrics
        metrics_history = [line for line in lines if line.get('event') == 'METRICS']
        
        all_runs_data.append({
            'experiment_id': config['experiment_id'],
            'is_failure': is_failure,
            'fault_type': config['fault_injection']['type'],
            'metrics_history': metrics_history
        })
    return all_runs_data

def fit_weibull_parameters(all_runs_data):
    """Fits Weibull parameters from the raw signal data in failing runs."""
    all_signals = {'lambda': [], 'sigma_sq': [], 'delta_l': []}
    for run in all_runs_data:
        if not run['is_failure']: continue
        for metrics in run['metrics_history']:
            if metrics['lambda'] > 1e-6: all_signals['lambda'].append(metrics['lambda'])
            if metrics['sigma_sq'] > 1e-6: all_signals['sigma_sq'].append(metrics['sigma_sq'])
            if metrics['delta_l'] > 1e-6: all_signals['delta_l'].append(metrics['delta_l'])
            
    fitted_params = {}
    for signal_name, data in all_signals.items():
        if not data:
            eta, beta = 1.0, 1.0 # Default values
        else:
            shape_beta, _, scale_eta = weibull_min.fit(data, floc=0)
        fitted_params[signal_name] = {'eta': scale_eta, 'beta': shape_beta}
    return fitted_params

def get_weibull_term(val, eta, beta):
    if val <= 0: return 0.0
    return (val / eta) ** beta

def analyze_performance(all_runs_data, fitted_params, alert_threshold):
    """Analyzes performance for a given set of parameters and threshold."""
    run_summaries = []
    for run in all_runs_data:
        alert_step = None
        for metrics in run['metrics_history']:
            exponent = sum([
                get_weibull_term(metrics['lambda'], fitted_params['lambda']['eta'], fitted_params['lambda']['beta']),
                get_weibull_term(metrics['sigma_sq'], fitted_params['sigma_sq']['eta'], fitted_params['sigma_sq']['beta']),
                get_weibull_term(metrics['delta_l'], fitted_params['delta_l']['eta'], fitted_params['delta_l']['beta'])
            ])
            r_value = np.exp(-exponent)
            if r_value < alert_threshold and alert_step is None:
                alert_step = metrics['step']
                break
        
        last_step = run['metrics_history'][-1]['step'] if run['metrics_history'] else 0
        run_summaries.append({
            'fault_type': run['fault_type'],
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
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr}, df

def main():
    print("--- Parsing all 108 log files ---")
    all_runs_data = parse_all_logs(LOG_DIRECTORY)
    
    print("\n--- Fitting Weibull parameters from data ---")
    base_fitted_params = fit_weibull_parameters(all_runs_data)
    for signal, params in base_fitted_params.items():
        print(f"Fitted params for '{signal}':\teta (scale) = {params['eta']:.4f}, beta (shape) = {params['beta']:.4f}")

    print("\n" + "="*60)
    print("--- Section 1: Core Performance Results ---")
    print("="*60)
    
    core_metrics, results_df = analyze_performance(all_runs_data, base_fitted_params, R_ALERT_THRESHOLD)
    
    # 1. Statistical Summary (Lead Time)
    detected_failures = results_df[(results_df['is_failure'] == True) & (results_df['alert_step'].notna())].copy()
    detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['alert_step']
    detected_failures['lead_time_minutes'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0
    
    print("\n1. Statistical Summary (Lead Time):")
    print(f"  - Mean Lead Time:         {detected_failures['lead_time_minutes'].mean():.2f} minutes")
    print(f"  - Median Lead Time:       {detected_failures['lead_time_minutes'].median():.2f} minutes")
    print(f"  - 95th Percentile:        {detected_failures['lead_time_minutes'].quantile(0.95):.2f} minutes")
    
    # Cohen's d calculation
    mean_diff = detected_failures['lead_time_minutes'].mean() - 0.5 # Baseline mean is ~0.5
    pooled_std = detected_failures['lead_time_minutes'].std()
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    print(f"  - Effect Size (Cohen's d): {cohens_d:.2f}")

    # 2. Cross-scenarios Analysis (Lead Time)
    print("\n2. Cross-scenarios Analysis (Lead Time by Fault Type):")
    lead_time_by_type = detected_failures.groupby('fault_type')['lead_time_minutes'].mean()
    print(lead_time_by_type.round(2).to_string())

    # 3. Classification Result
    print("\n3. Classification Result:")
    print(f"  - Precision:              {core_metrics['precision']:.3f}")
    print(f"  - Recall:                 {core_metrics['recall']:.3f}")
    print(f"  - F1-Score:               {core_metrics['f1']:.3f}")
    print(f"  - False Positive Rate:    {core_metrics['fpr']:.3f}")
    # Note: ROC-AUC requires scores, not just binary predictions. We'll simulate this.
    # For the paper, you'd plot the ROC curve based on varying the threshold.
    # Here we just report the F1 for the chosen threshold.

    print("\n" + "="*60)
    print("--- Section 2: Sensitivity & Robustness (for Table 4) ---")
    print("="*60)
    
    # 5 & 6. Parameter Sensitivity and Threshold Robustness
    print("\n5 & 6. Sensitivity and Robustness Analysis (F1-Score):")
    print(f"{'Parameter':<25} | {'-20%':<10} | {'-10%':<10} | {'Baseline':<10} | {'+10%':<10} | {'+20%':<10}")
    print("-" * 80)

    # Threshold Robustness
    f1_scores = []
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        metrics, _ = analyze_performance(all_runs_data, base_fitted_params, R_ALERT_THRESHOLD * multiplier)
        f1_scores.append(f"{metrics['f1']:.3f}")
    print(f"{'R Alert Threshold':<25} | {f1_scores[0]:<10} | {f1_scores[1]:<10} | {f1_scores[2]:<10} | {f1_scores[3]:<10} | {f1_scores[4]:<10}")

    # Eta (scale) Sensitivity
    f1_scores = []
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        perturbed_params = {k: v.copy() for k, v in base_fitted_params.items()}
        for signal in perturbed_params: perturbed_params[signal]['eta'] *= multiplier
        metrics, _ = analyze_performance(all_runs_data, perturbed_params, R_ALERT_THRESHOLD)
        f1_scores.append(f"{metrics['f1']:.3f}")
    print(f"{'All η (scale) params.':<25} | {f1_scores[0]:<10} | {f1_scores[1]:<10} | {f1_scores[2]:<10} | {f1_scores[3]:<10} | {f1_scores[4]:<10}")

    # Beta (shape) Sensitivity
    f1_scores = []
    for multiplier in [0.8, 0.9, 1.0, 1.1, 1.2]:
        perturbed_params = {k: v.copy() for k, v in base_fitted_params.items()}
        for signal in perturbed_params: perturbed_params[signal]['beta'] *= multiplier
        metrics, _ = analyze_performance(all_runs_data, perturbed_params, R_ALERT_THRESHOLD)
        f1_scores.append(f"{metrics['f1']:.3f}")
    print(f"{'All β (shape) params.':<25} | {f1_scores[0]:<10} | {f1_scores[1]:<10} | {f1_scores[2]:<10} | {f1_scores[3]:<10} | {f1_scores[4]:<10}")


if __name__ == "__main__":
    main()
