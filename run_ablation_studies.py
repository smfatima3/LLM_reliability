import itertools
import copy
import os
from simulator import TrainingSimulator, CONFIG as BASE_CONFIG

def main():
    """
    This script systematically runs experiments for each ablation mode and
    ensures the simulator saves logs into separate, clearly named subdirectories.
    """
    MODELS = ['GPT-2', 'T5', 'BERT']
    REPETITIONS = [1, 2, 3, 4, 5, 6, 7, 8]
    
    # We test all fault types for each mode to ensure a fair comparison
    FAULT_TYPES = ['GRADIENT_EXPLOSION', 'LR_SPIKE', 'DATA_CORRUPTION', 'NONE']
    
    # The different modes we want to test
    ABLATION_MODES = ['FULL_METRIC', 'LAMBDA_ONLY', 'SIGMA_ONLY', 'DELTA_L_ONLY']

    # Define the configurations for each mode
    mode_configs = {
        'FULL_METRIC': {'METRIC_MODE': 'FULL_METRIC', 'r_metric_weights': {'w_sigma': 0.5, 'w_delta': 0.78}, 'alert_thresholds': {'r_metric': 0.57}},
        'LAMBDA_ONLY': {'METRIC_MODE': 'LAMBDA_ONLY', 'alert_thresholds': {'r_metric': 0.57}},
        'SIGMA_ONLY': {'METRIC_MODE': 'SIGMA_ONLY', 'alert_thresholds': {'r_metric': 0.57}},
        'DELTA_L_ONLY': {'METRIC_MODE': 'DELTA_L_ONLY', 'alert_thresholds': {'r_metric': 0.57}}
    }

    for mode in ABLATION_MODES:
        print("\n" + "="*25 + f" RUNNING ABLATION MODE: {mode} " + "="*25)
        experiment_counter = 0
        
        # This will create 3 models * 4 fault types * 3 reps = 36 runs per mode
        for model, fault_type, rep in itertools.product(MODELS, FAULT_TYPES, REPETITIONS):
            # Create a fresh config for each run
            config = copy.deepcopy(BASE_CONFIG)
            
            # Update config with mode-specific settings from our dictionary
            config.update(mode_configs[mode])
            
            # Set the unique identifiers for this specific run
            config['experiment_id'] = f"run_{experiment_counter:03d}_{model.lower()}_{fault_type.lower()}_rep{rep}"
            config['model_type'] = model
            config['fault_injection']['type'] = fault_type
            
            print(f"\n--- Running Experiment {experiment_counter+1}: {config['experiment_id']} ---")
            
            try:
                # Initialize and run the simulation with the fully prepared config
                simulator = TrainingSimulator(config)
                simulator.run()
            except Exception as e:
                print(f"!!! Experiment {config['experiment_id']} failed with an error: {e} !!!")
            
            experiment_counter += 1

    print("\n--- All Ablation Studies Completed ---")

if __name__ == '__main__':
    main()
