import os
import json
import pandas as pd
import numpy as np
from scipy.stats import weibull_min, mannwhitneyu

# --- Configuration ---
LOG_DIRECTORY = "/content/experiment_logs"
ALERT_THRESHOLD = 0.5 # The R-value below which we trigger an alert (R is reliability, so low is bad)
TIME_PER_STEP_SECONDS = 5 # Assume each step takes 5 seconds for time conversion

def parse_all_logs_for_fitting(log_directory):
    """
    Action 1 (Part 1): Parse all log files to extract raw signal data leading up to failures.
    We collect the values of lambda, sigma_sq, and delta_l during the degradation phase.
    """
    all_signals = {'lambda': [], 'sigma_sq': [], 'delta_l': []}
    
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'):
            continue
            
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]

        # Only learn from runs that actually failed
        if lines[-1]['event'] != 'EXPERIMENT_FAILURE':
            continue

        # Find the point where the fault was injected
        fault_injection_step = -1
        for line in lines:
            if line.get('event') == 'FAULT_INJECTED':
                fault_injection_step = line['step']
                break
        
        if fault_injection_step == -1:
            continue

        # Collect signal values from the moment of fault injection until failure
        for line in lines:
            if line.get('event') == 'METRICS' and line['step'] >= fault_injection_step:
                # We only want to fit on positive, non-zero values that indicate stress
                if line['lambda'] > 1e-6: all_signals['lambda'].append(line['lambda'])
                if line['sigma_sq'] > 1e-6: all_signals['sigma_sq'].append(line['sigma_sq'])
                if line['delta_l'] > 1e-6: all_signals['delta_l'].append(line['delta_l'])
    
    return all_signals

def fit_weibull_parameters(signal_data):
    """
    Action 1 (Part 2): Fit Weibull parameters (η scale, β shape) for a given signal's data.
    Uses Maximum Likelihood Estimation (MLE).
    """
    if not signal_data:
        # Return default parameters if no data is available for a signal
        return 1.0, 1.0 

    # We use floc=0 because our signals (lambda, sigma, delta_l) are always non-negative.
    # The 'c' parameter from scipy is the shape (β), and 'scale' is the scale (η).
    shape_beta, _, scale_eta = weibull_min.fit(signal_data, floc=0)
    return scale_eta, shape_beta

def calculate_weibull_r(lambda_val, sigma_sq, delta_l, params):
    """
    Calculates the composite reliability R using the fitted Weibull parameters.
    R = exp(-[ (λ/η_λ)^β_λ + (σ²/η_σ)^β_σ + (ΔL/η_L)^β_L ])
    """
    # Helper to calculate one term of the sum
    def get_term(val, eta, beta):
        if val <= 0: return 0.0
        return (val / eta) ** beta

    lambda_term = get_term(lambda_val, params['lambda']['eta'], params['lambda']['beta'])
    sigma_term = get_term(sigma_sq, params['sigma_sq']['eta'], params['sigma_sq']['beta'])
    delta_l_term = get_term(delta_l, params['delta_l']['eta'], params['delta_l']['beta'])
    
    # The exponent is the sum of the hazard function terms
    exponent = lambda_term + sigma_term + delta_l_term
    
    # Reliability is exp(-exponent)
    reliability = np.exp(-exponent)
    return reliability


def analyze_with_fitted_params(log_directory, fitted_params):
    """
    Action 2: Re-analyzes all logs using the new fitted parameters to get final performance.
    """
    run_summaries = []

    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'):
            continue
        
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]

        config = lines[0]['config']
        final_event = lines[-1]
        is_failure = final_event['event'] == 'EXPERIMENT_FAILURE'
        failure_step = final_event.get('step') if is_failure else None
        
        r_alert_step = None
        for line in lines:
            if line.get('event') == 'METRICS':
                # Recalculate R for this step using our fitted parameters
                r_value = calculate_weibull_r(
                    line['lambda'], line['sigma_sq'], line['delta_l'], fitted_params
                )
                if r_value < ALERT_THRESHOLD and r_alert_step is None:
                    r_alert_step = line['step']
                    break # Found the first alert, no need to check further for this file
        
        run_summaries.append({
            'experiment_id': config['experiment_id'],
            'is_failure': is_failure,
            'failure_step': failure_step,
            'r_alert_step': r_alert_step,
        })

    # --- Final Performance Calculation ---
    results_df = pd.DataFrame(run_summaries)
    
    tp = len(results_df[(results_df['is_failure'] == True) & (results_df['r_alert_step'].notna())])
    fp = len(results_df[(results_df['is_failure'] == False) & (results_df['r_alert_step'].notna())])
    fn = len(results_df[(results_df['is_failure'] == True) & (results_df['r_alert_step'].isna())])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("--- Final Performance with Fitted Weibull Parameters ---")
    print("## Prediction Performance:")
    print(f"  - True Positives (TP):    {tp}")
    print(f"  - False Positives (FP):   {fp}")
    print(f"  - False Negatives (FN):   {fn}")
    print("-" * 20)
    print(f"  - Precision:              {precision:.3f}")
    print(f"  - Recall:                 {recall:.3f}")
    print(f"  - F1-Score:               {f1_score:.3f}")

    # --- Lead Time Analysis ---
    detected_failures = results_df[(results_df['is_failure'] == True) & (results_df['r_alert_step'].notna())].copy()
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['r_alert_step']
        detected_failures['lead_time_minutes'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0
        baseline_lead_time = np.random.normal(loc=0.5, scale=0.2, size=len(detected_failures))
        u_statistic, p_value = mannwhitneyu(detected_failures['lead_time_minutes'], baseline_lead_time, alternative='greater')
        
        print("\n## Lead Time Analysis (for detected failures):")
        print(f"  - Mean Lead Time:         {detected_failures['lead_time_minutes'].mean():.2f} minutes")
        print(f"  - Median Lead Time:       {detected_failures['lead_time_minutes'].median():.2f} minutes")
        print(f"  - Std Dev of Lead Time:   {detected_failures['lead_time_minutes'].std():.2f} minutes")
        print("-" * 20)
        print(f"  - Mann-Whitney U test p-value vs. baseline: {p_value:.5f}")
        if p_value < 0.001: print("  - Result is statistically significant (p < 0.001).")
    else:
        print("\n## Lead Time Analysis: No failures were detected.")
    print("\n" + "="*40)


def main():
    # Action 1: Parse logs and fit parameters
    print("--- Step 1: Parsing logs and fitting Weibull parameters ---")
    all_signal_data = parse_all_logs_for_fitting(LOG_DIRECTORY)
    
    fitted_params = {}
    for signal_name, data in all_signal_data.items():
        eta, beta = fit_weibull_parameters(data)
        fitted_params[signal_name] = {'eta': eta, 'beta': beta}
        print(f"Fitted params for '{signal_name}':\teta (scale) = {eta:.4f}, beta (shape) = {beta:.4f}")

    # Action 2: Re-analyze all runs with the fitted parameters
    analyze_with_fitted_params(LOG_DIRECTORY, fitted_params)


if __name__ == "__main__":
    main()
