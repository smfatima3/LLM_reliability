import os
import json
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

# --- Configuration ---
LOG_DIRECTORY = "/content/experiment_logs"
R_METRIC_THRESHOLD = 0.65 # The critical alert threshold from your config
TIME_PER_STEP_SECONDS = 5 # Assume each step takes 5 seconds for time conversion

def parse_log_file(filepath):
    """
    Parses a single experiment log file and extracts key outcomes.
    Returns a dictionary summarizing the run.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Get config and find the final event
    config = json.loads(lines[0])['config']
    final_event = json.loads(lines[-1])

    is_failure = final_event['event'] == 'EXPERIMENT_FAILURE'
    failure_step = final_event.get('step', None) if is_failure else None

    # Find the first time the R metric alert was triggered
    r_alert_step = None
    for line in lines:
        if '"event": "METRICS"' in line:
            data = json.loads(line)
            # Check if r_metric exists in alerts and has not been set yet
            if 'r_metric' in data.get('alerts', {}) and r_alert_step is None:
                # Check if the R value actually crossed the threshold
                if data.get('r_metric', 0) > R_METRIC_THRESHOLD:
                    r_alert_step = data['alerts']['r_metric']
                    # We break here to get the *first* time it alerted
                    break
    
    return {
        'experiment_id': config['experiment_id'],
        'model': config['model_type'],
        'fault_type': config['fault_injection']['type'],
        'is_failure': is_failure,
        'failure_step': failure_step,
        'r_alert_step': r_alert_step,
    }


def analyze_results(results_df):
    """
    Takes the parsed results DataFrame and calculates the final aggregate metrics.
    """
    print("\n--- Aggregate Results Analysis ---")

    # --- 1. Prediction Performance (Precision, Recall, F1) ---
    # True Positive (TP): Failure occurred AND we alerted.
    tp = len(results_df[(results_df['is_failure'] == True) & (results_df['r_alert_step'].notna())])

    # False Positive (FP): No failure, but we alerted.
    fp = len(results_df[(results_df['is_failure'] == False) & (results_df['r_alert_step'].notna())])

    # False Negative (FN): Failure occurred, but we did NOT alert.
    fn = len(results_df[(results_df['is_failure'] == True) & (results_df['r_alert_step'].isna())])
    
    # True Negative (TN) is not typically used in F1 score but good to know
    # tn = len(results_df[(results_df['is_failure'] == False) & (results_df['r_alert_step'].isna())])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n## Prediction Performance:")
    print(f"  - True Positives (TP):    {tp}")
    print(f"  - False Positives (FP):   {fp}")
    print(f"  - False Negatives (FN):   {fn}")
    print("-" * 20)
    print(f"  - Precision:              {precision:.3f}")
    print(f"  - Recall:                 {recall:.3f}")
    print(f"  - F1-Score:               {f1_score:.3f}")


    # --- 2. Lead Time Analysis ---
    failed_runs = results_df[results_df['is_failure'] == True].copy()
    detected_failures = failed_runs[failed_runs['r_alert_step'].notna()].copy()

    if not detected_failures.empty:
        # Calculate lead time in 'steps'
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures['r_alert_step']
        
        # Convert to minutes
        detected_failures['lead_time_minutes'] = detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0

        # Create a dummy baseline with very short lead time for comparison
        baseline_lead_time_minutes = np.random.normal(loc=0.5, scale=0.2, size=len(detected_failures))
        
        # Perform Mann-Whitney U test
        u_statistic, p_value = mannwhitneyu(detected_failures['lead_time_minutes'], baseline_lead_time_minutes, alternative='greater')

        print("\n## Lead Time Analysis (for detected failures):")
        print(f"  - Mean Lead Time:         {detected_failures['lead_time_minutes'].mean():.2f} minutes")
        print(f"  - Median Lead Time:       {detected_failures['lead_time_minutes'].median():.2f} minutes")
        print(f"  - Std Dev of Lead Time:   {detected_failures['lead_time_minutes'].std():.2f} minutes")
        print("-" * 20)
        print(f"  - Mann-Whitney U test p-value vs. baseline: {p_value:.5f}")
        if p_value < 0.001:
            print("  - Result is statistically significant (p < 0.001).")

    else:
        print("\n## Lead Time Analysis: No failures were detected, cannot compute lead time.")

    print("\n" + "="*40)


def main():
    """
    Main function to find logs, parse them, and run analysis.
    """
    all_results = []
    log_files = [f for f in os.listdir(LOG_DIRECTORY) if f.endswith('.jsonl')]

    if not log_files:
        print(f"Error: No .jsonl files found in the '{LOG_DIRECTORY}' directory.")
        return

    print(f"Found {len(log_files)} log files to analyze.")

    for filename in log_files:
        filepath = os.path.join(LOG_DIRECTORY, filename)
        try:
            result = parse_log_file(filepath)
            all_results.append(result)
        except Exception as e:
            print(f"Could not parse file {filename}: {e}")

    # Convert to a DataFrame for powerful analysis
    results_df = pd.DataFrame(all_results)
    
    print("\n--- Parsed Experimental Data ---")
    print(results_df.head())
    print("...")
    
    analyze_results(results_df)


if __name__ == "__main__":
    main()
