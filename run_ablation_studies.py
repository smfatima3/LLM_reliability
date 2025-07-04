import itertools
import copy
import os
from simulator import TrainingSimulator, CONFIG as BASE_CONFIG

def main():
    """
    This script systematically runs experiments for each ablation mode using 
    modern architectures and fault types. It ensures the simulator saves 
    logs into separate, clearly named subdirectories for comprehensive analysis.
    """
    # 1. Updated with Modern Architectures
    MODELS = ['Llama-3-8B', 'Mistral-7B', 'GPT-4-MoE']
    REPETITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Reduced repetitions for a quicker example run
    
    # 2. Updated with Modern and Classic Fault Types for broad coverage
    FAULT_TYPES = [
        'EXPERT_FAILURE',        # MoE Specific
        'ROUTER_IMBALANCE',      # MoE Specific
        'GRADIENT_EXPLOSION',    # Classic model instability
        'LR_SPIKE',              # Classic model instability
        'NONE'                   # Control group
    ]
    
    # 3. All modes will be run to gather data for a full baseline comparison
    ABLATION_MODES = ['FULL_METRIC', 'LAMBDA_ONLY', 'SIGMA_ONLY', 'DELTA_L_ONLY']

    # Define the configurations for each mode
    # Note: LAMBDA_ONLY is not represented in the simplified simulator but is kept for structure.
    mode_configs = {
        'FULL_METRIC': {'METRIC_MODE': 'FULL_METRIC'},
        'LAMBDA_ONLY': {'METRIC_MODE': 'LAMBDA_ONLY'},
        'SIGMA_ONLY': {'METRIC_MODE': 'SIGMA_ONLY'},
        'DELTA_L_ONLY': {'METRIC_MODE': 'DELTA_L_ONLY'}
    }

    total_runs = len(MODELS) * len(FAULT_TYPES) * len(REPETITIONS) * len(ABLATION_MODES)
    print(f"--- Starting Ablation Study with Modern Architectures ---")
    print(f"Total experiments to run: {total_runs}")

    run_counter = 0
    for mode in ABLATION_MODES:
        print("\n" + "="*25 + f" RUNNING ABLATION MODE: {mode} " + "="*25)
        
        for model, fault_type, rep in itertools.product(MODELS, FAULT_TYPES, REPETITIONS):
            run_counter += 1
            # Create a fresh config for each run
            config = copy.deepcopy(BASE_CONFIG)
            
            # Update config with mode-specific settings
            config.update(mode_configs[mode])
            
            # Set unique identifiers
            config['experiment_id'] = f"run_{run_counter:03d}_{model.lower()}_{fault_type.lower()}_rep{rep}"
            config['model_type'] = model
            config['fault_injection']['type'] = fault_type
            
            # Set fault-specific parameters
            if fault_type == 'EXPERT_FAILURE':
                config['fault_injection']['params'] = {'expert_to_kill': 2}
            elif fault_type == 'ROUTER_IMBALANCE':
                config['fault_injection']['params'] = {'imbalance_factor': 0.7}
            
            print(f"\n--- Running Experiment {run_counter}/{total_runs}: {config['experiment_id']} ---")
            
            try:
                # Initialize and run the simulation
                simulator = TrainingSimulator(config)
                simulator.run()
            except Exception as e:
                print(f"!!! Experiment {config['experiment_id']} failed with an error: {e} !!!")

    print("\n--- All Ablation Studies Completed ---")

if __name__ == '__main__':
    main()
