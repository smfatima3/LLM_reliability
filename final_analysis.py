import os, json, pandas as pd

def analyze_directory(log_directory):
    run_summaries = []
    if not os.path.exists(log_directory): return pd.DataFrame()
    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f: lines = [json.loads(line) for line in f]
        is_failure = lines[-1]['event'] == 'EXPERIMENT_FAILURE'
        alert_step = next((line['alerts']['r_metric'] for line in lines if line.get('alerts', {}).get('r_metric')), None)
        run_summaries.append({'is_failure': is_failure, 'alert_step': alert_step})
    return pd.DataFrame(run_summaries)

def print_performance_report(df, method_name):
    tp = len(df[(df['is_failure'] == True) & (df['alert_step'].notna())])
    fp = len(df[(df['is_failure'] == False) & (df['alert_step'].notna())])
    fn = len(df[(df['is_failure'] == True) & (df['alert_step'].isna())])
    tn = len(df[(df['is_failure'] == False) & (df['alert_step'].isna())])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    print(f"| {method_name:<20} | {precision:^9.3f} | {recall:^8.3f} | {f1:^10.3f} | {fpr:^7.3f} |")

# You would run this analysis script for each subdirectory of logs
# Example:
df_delta_l = analyze_directory('logs_DELTA_L_ONLY')
print_performance_report(df_delta_l, 'DELTA_L_ONLY')
