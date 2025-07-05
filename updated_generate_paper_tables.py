import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# --- Configuration ---
LOG_PARENT_DIRECTORY = "." 
LOG_MODE_TO_ANALYZE = 'FULL_METRIC' 
FINAL_R_METRIC_THRESHOLD = 0.57
TIME_PER_STEP_SECONDS = 5

# --- Helper Functions ---

def parse_and_analyze_logs(log_mode):
    """
    Parses logs and calculates alert steps for the R-Metric AND the new
    Isolation Forest baseline.
    """
    log_directory = os.path.join(LOG_PARENT_DIRECTORY, f"logs_{log_mode}")
    
    if not os.path.exists(log_directory):
        print(f"FATAL ERROR: Log directory not found at '{log_directory}'.")
        return pd.DataFrame()

    run_summaries = []
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'):
            continue
        
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            lines = [json.loads(line) for line in f]
        
        if not lines:
            continue

        config = lines[0].get('config', {})
        final_event = lines[-1]
        
        metrics_df = pd.DataFrame([line for line in lines if line.get('event') == 'METRICS'])
        if metrics_df.empty:
            continue

        summary = {
            'is_failure': final_event.get('event') == 'EXPERIMENT_FAILURE',
            'failure_step': final_event.get('step') if final_event.get('event') == 'EXPERIMENT_FAILURE' else None,
            'fault_type': config.get('fault_injection', {}).get('type', 'NONE'),
            'r_metric_alert_step': None,
            'isolation_forest_alert_step': None, # New baseline
        }

        # Calculate R-Metric alert step
        alert_df = metrics_df[metrics_df['r_metric'] > FINAL_R_METRIC_THRESHOLD]
        if not alert_df.empty:
            summary['r_metric_alert_step'] = alert_df['step'].min()

        # --- New: Calculate Isolation Forest alert step ---
        if 'validation_loss' in metrics_df.columns and len(metrics_df) > 1:
            # Reshape data for sklearn: needs to be 2D array
            loss_data = metrics_df['validation_loss'].values.reshape(-1, 1)
            
            # Fit the model and get predictions (-1 for anomalies, 1 for inliers)
            iso_forest = IsolationForest(contamination='auto', random_state=42)
            predictions = iso_forest.fit_predict(loss_data)
            
            # Find the first anomaly detected by the model
            anomaly_indices = np.where(predictions == -1)[0]
            if len(anomaly_indices) > 0:
                first_anomaly_index = anomaly_indices[0]
                summary['isolation_forest_alert_step'] = metrics_df.iloc[first_anomaly_index]['step']
        # ----------------------------------------------------

        run_summaries.append(summary)
        
    return pd.DataFrame(run_summaries)

def calculate_performance(df, alert_col_name):
    """Calculates key performance indicators for a given alert method."""
    if df.empty or alert_col_name not in df.columns:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'lead_time': 0}

    tp = len(df[(df['is_failure'] == True) & (df[alert_col_name].notna())])
    fp = len(df[(df['is_failure'] == False) & (df[alert_col_name].notna())])
    fn = len(df[(df['is_failure'] == True) & (df[alert_col_name].isna())])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    detected_failures = df[(df['is_failure'] == True) & (df[alert_col_name].notna())].copy()
    lead_time = 0
    if not detected_failures.empty:
        detected_failures['lead_time_steps'] = detected_failures['failure_step'] - detected_failures[alert_col_name]
        lead_time = (detected_failures['lead_time_steps'] * TIME_PER_STEP_SECONDS / 60.0).mean()

    return {'precision': precision, 'recall': recall, 'f1': f1, 'lead_time': lead_time}

def generate_table_3_baseline_comparison(results_df):
    """Generates Table 3, now including the Isolation Forest baseline."""
    print("\n" + "="*80)
    print("### Table 3: Comparison with Baselines (with Isolation Forest) ###")
    print("="*80)
    print(f"| {'Method':<32} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'Lead Time (min)':^17} |")
    print(f"|{'-'*34}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*19}|")
    
    baselines = {
        "Our R-Metric (Full)": "r_metric_alert_step",
        "Isolation Forest (Val. Loss)": "isolation_forest_alert_step", # New baseline
    }
    
    for name, col in baselines.items():
        perf = calculate_performance(results_df, col)
        print(f"| {name:<32} | {perf['precision']:^9.3f} | {perf['recall']:^8.3f} | {perf['f1']:^10.3f} | {perf['lead_time']:^17.1f} |")
    
    print("="*80)


def main():
    """
    Main function to orchestrate parsing logs and generating all tables.
    """
    print("--- Starting Paper Table Generation Script ---")
    
    results_df = parse_and_analyze_logs(LOG_MODE_TO_ANALYZE)
    
    if results_df.empty:
        print("\nNo data to analyze. Script will now exit.")
        return
        
    print(f"Analysis complete. Found {len(results_df)} experimental runs.")
    
    # Generate the updated Table 3
    generate_table_3_baseline_comparison(results_df)
    
    print("\n--- All tables generated successfully. ---")

if __name__ == "__main__":
    main()
