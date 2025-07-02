import os
import json
import pandas as pd

def analyze_directory(log_directory):
    """Analyzes all logs in a single directory to get performance stats."""
    run_summaries = []
    if not os.path.exists(log_directory):
        return {'precision': 0, 'recall': 0, 'f1': 0, 'fpr': 0}

    for filename in os.listdir(log_directory):
        if not filename.endswith('.jsonl'): continue
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f: lines = [json.loads(line) for line in f]
        
        is_failure = lines[-1]['event'] == 'EXPERIMENT_FAILURE'
        alert_step = None
        for line in lines:
            if line.get('event') == 'METRICS' and line.get('alerts', {}).get('r_metric'):
                alert_step = line['alerts']['r_metric']
                break
        
        run_summaries.append({'is_failure': is_failure, 'alert_step': alert_step})

    df = pd.DataFrame(run_summaries)
    tp = len(df[(df['is_failure'] == True) & (df['alert_step'].notna())])
    fp = len(df[(df['is_failure'] == False) & (df['alert_step'].notna())])
    fn = len(df[(df['is_failure'] == True) & (df['alert_step'].isna())])
    tn = len(df[(df['is_failure'] == False) & (df['alert_step'].isna())])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'fpr': fpr}

def main():
    ABLATION_MODES = ['FULL_METRIC', 'LAMBDA_ONLY', 'SIGMA_ONLY', 'DELTA_L_ONLY']
    
    print("\n" + "="*60)
    print("--- Ablation Study Results ---")
    print(f"| {'Method':<20} | {'Precision':^9} | {'Recall':^8} | {'F1-Score':^10} | {'FPR':^7} |")
    print(f"|{'-'*22}|{'-'*11}|{'-'*10}|{'-'*12}|{'-'*9}|")

    for mode in ABLATION_MODES:
        log_dir = f"logs_{mode}"
        results = analyze_directory(log_dir)
        print(f"| {mode:<20} | {results['precision']:^9.3f} | {results['recall']:^8.3f} | {results['f1']:^10.3f} | {results['fpr']:^7.3f} |")

    print("="*60)

if __name__ == '__main__':
    main()
