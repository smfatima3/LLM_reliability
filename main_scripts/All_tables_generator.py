import os
import json
import pandas as pd
import numpy as np

# --- Configuration ---
# Set the parent directory where your log folders are located.
LOG_PARENT_DIRECTORY = "." 
# We will analyze the logs from the full composite metric run.
LOG_MODE_TO_ANALYZE = 'FULL_METRIC' 

# --- Constants for Analysis ---
# These values should match the ones used in your final experiments.
FINAL_R_METRIC_THRESHOLD = 0.57
TIME_PER_STEP_SECONDS = 5 # Assumed time for one training step.

# --- Helper Functions ---

def parse_and_analyze_logs(log_mode):
    """
    Parses all log files for a given mode, calculates alert steps for the R-Metric
    and various baselines, and returns a comprehensive DataFrame.
    """
    log_directory = os.path.join(LOG_PARENT_DIRECTORY, f"logs_{log_mode}")
    
    if not os.path.exists(log_directory):
        print(f"FATAL ERROR: Log directory not found at '{log_directory}'.")
        print("Please ensure experiments have been run and the directory exists.")
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
        
        summary = {
            'is_failure': final_event.get('event') == 'EXPERIMENT_FAILURE',
            'failure_step': final_event.get('step') if final_event.get('event') == 'EXPERIMENT_FAILURE' else None,
            'fault_type': config.get('fault_injection', {}).get('type', 'NONE'),
            'r_metric_alert_step': None,
            'slope_baseline_alert_step': None,
            'sigma_only_alert_step': None,
            'delta_l_only_alert_step': None
        }

        # Analyze metric history to find first alert for each method
        loss_history = []
        for line in lines:
            if line.get('event') != 'METRICS':
                continue

            # 1. Our R-Metric
            if summary['r_metric_alert_step'] is None and line.get('r_metric', 0) > FINAL_R_METRIC_THRESHOLD:
                summary['r_metric_alert_step'] = line.get('step')

            # 2. Validation Loss Slope Baseline
            loss_history.append(line.get('validation_loss', 1e6))
            if summary['slope_baseline_alert_step'] is None and len(loss_history) >= 3:
                if loss_history[-1] > loss_history[-2] and loss_history[-2] > loss_history[-3]:
                    summary['slope_baseline_alert_step'] = line.get('step')
            
            # 3. Ablation Baselines (using normalized components as proxies)
            # We use the same threshold for a fair comparison of the signal's strength
            if summary['sigma_only_alert_step'] is None and line.get('sigma_sq_norm', 0) > FINAL_R_METRIC_THRESHOLD:
                summary['sigma_only_alert_step'] = line.get('step')

            if summary['delta_l_only_alert_step'] is None and line.get('delta_l_norm', 0) > FINAL_R_METRIC_THRESHOLD:
                summary['delta_l_only_alert_step'] = line.get('step')

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

# --- Table Generation Functions ---

def generate_table_1_comparison_approaches(our_metric_performance):
    """Generates Table 1: Comparison with existing approaches."""
    print("\n" + "="*80)
    print("### Table 1: Comparison with Existing Approaches ###")
    print("="*80)
    print(f"| {'Approach':<25} | {'Reactive/Proactive':<20} | {'Scope':<15} | {'Lead Time (min)':<18} |")
    print(f"|{'-'*27}|{'-'*22}|{'-'*17}|{'-'*20}|")
    
    # Static data for existing approaches
    print(f"| {'Checkpointing':<25} | {'Reactive':<20} | {'System State':<15} | {'0':<18} |")
    print(f"| {'Simple Heuristics':<25} | {'Reactive':<20} | {'Single Signal':<15} | {'~2-5':<18} |")
    
    # Dynamic data for our metric
    our_lead_time = f"{our_metric_performance['lead_time']:.1f}"
    print(f"| {'Our R-Metric (Full)':<25} | {'Proactive':<20} | {'Integrated Health':<15} | {our_lead_time:<18} |")
    print("="*80)

def generate_table_2_experimental_matrix():
    """Generates Table 2: The static experimental scenarios matrix from your plan."""
    print("\n" + "="*80)
    print("### Table 2: Experimental Scenarios Matrix ###")
    print("="*80)
    print(f"| {'Scenario':<10} | {'Model Architecture':<20} | {'Hardware Faults':<20} | {'Model Instability':<25} |")
    print(f"|{'-'*12}|{'-'*22}|{'-'*22}|{'-'*27}|")
    print(f"| {'S1':<10} | {'Llama-3-8B':<20} | {'Low (Simulated)':<20} | {'None':<25} |")
    print(f"| {'S2':<10} | {'GPT-4-MoE':<20} | {'High (Simulated)':<20} | {'EXPERT_FAILURE':<25} |")
    print(f"| {'S3':<10} | {'Mistral-7B':<20} | {'Medium (Simulated)':<20} | {'GQA_MISMATCH':<25} |")
    print(f"| {'S4':<10} | {'Llama-3-8B':<20} | {'Low (Simulated)':<20} | {'ROUTER_IMBALANCE':<25} |")
    print("="*80)

def generate_table_3_baseline_comparison(results_df):
    """Generates Table 3: A direct comparison of our metric vs. baselines."""
    print("\n" + "="*80)
    print("### Table 3: Comparison with Baselines ###")
    print("="*80)
    print(f"| {'Method':<28} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'Lead Time (min)':^17} |")
    print(f"|{'-'*30}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*19}|")
    
    baselines = {
        "Our R-Metric (Full)": "r_metric_alert_step",
        "Val. Loss Slope Baseline": "slope_baseline_alert_step",
        "Ablation: σ² only": "sigma_only_alert_step",
        "Ablation: ΔL only": "delta_l_only_alert_step"
    }
    
    for name, col in baselines.items():
        perf = calculate_performance(results_df, col)
        print(f"| {name:<28} | {perf['precision']:^9.3f} | {perf['recall']:^8.3f} | {perf['f1']:^10.3f} | {perf['lead_time']:^17.1f} |")
    
    print("="*80)

def generate_table_4_performance_by_failure_type(results_df):
    """Generates Table 4: Performance breakdown by the type of injected fault."""
    print("\n" + "="*80)
    print("### Table 4: Prediction Performance by Failure Type ###")
    print("="*80)
    print(f"| {'Failure Type':<28} | {'Failure Count':^14} | {'Precision':^11} | {'Recall':^8} | {'F1-Score':^10} |")
    print(f"|{'-'*30}|{'-'*16}|{'-'*13}|{'-'*10}|{'-'*12}|")
    
    fault_runs_df = results_df[results_df['fault_type'] != 'NONE']
    
    # Overall Performance
    overall_perf = calculate_performance(fault_runs_df, 'r_metric_alert_step')
    total_failures = len(fault_runs_df[fault_runs_df['is_failure']])
    print(f"| {'Overall':<28} | {total_failures:^14} | {overall_perf['precision']:^11.3f} | {overall_perf['recall']:^8.3f} | {overall_perf['f1']:^10.3f} |")
    print(f"|{'-'*30}|{'-'*16}|{'-'*13}|{'-'*10}|{'-'*12}|")

    # Per-Type Performance
    for fault_type in sorted(fault_runs_df['fault_type'].unique()):
        df_subset = fault_runs_df[fault_runs_df['fault_type'] == fault_type]
        if not df_subset.empty:
            perf = calculate_performance(df_subset, 'r_metric_alert_step')
            failure_count = len(df_subset[df_subset['is_failure']])
            print(f"| {fault_type:<28} | {failure_count:^14} | {perf['precision']:^11.3f} | {perf['recall']:^8.3f} | {perf['f1']:^10.3f} |")

    print("="*80)

def main():
    """
    Main function to orchestrate parsing logs and generating all tables.
    """
    print("--- Starting Paper Table Generation Script ---")
    
    # Step 1: Parse logs and perform the core analysis once.
    print(f"Analyzing logs from 'logs_{LOG_MODE_TO_ANALYZE}' directory...")
    results_df = parse_and_analyze_logs(LOG_MODE_TO_ANALYZE)
    
    if results_df.empty:
        print("\nNo data to analyze. Script will now exit.")
        return
        
    print(f"Analysis complete. Found {len(results_df)} experimental runs.")
    
    # Step 2: Generate each table using the analyzed data.
    our_metric_performance = calculate_performance(results_df, 'r_metric_alert_step')

    generate_table_1_comparison_approaches(our_metric_performance)
    generate_table_2_experimental_matrix()
    generate_table_3_baseline_comparison(results_df)
    generate_table_4_performance_by_failure_type(results_df)
    
    print("\n--- All tables generated successfully. ---")

if __name__ == "__main__":
    main()
