import itertools
import copy
from simulator import TrainingSimulator, CONFIG as BASE_CONFIG

# --- Step 1: Define the Experiment Matrix ---

MODELS = ['GPT-2', 'T5', 'BERT']
REPETITIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Use seeds or just repetition counts

# Faults with severity levels
SEVERITY_FAULTS = {
    'LR_SPIKE': {
        'low': {'lr_multiplier': 5.0},
        'medium': {'lr_multiplier': 15.0},
        'high': {'lr_multiplier': 25.0}
    },
    'NETWORK_DEGRADATION': {
        'low': {'delay_ms': 50},
        'medium': {'delay_ms': 150},
        'high': {'delay_ms': 250}
    },
    'DATA_CORRUPTION': {
        'low': {'corruption_rate': 0.02},
        'medium': {'corruption_rate': 0.05},
        'high': {'corruption_rate': 0.10}
    }
}

# Faults without severity levels
SIMPLE_FAULTS = ['NODE_FAILURE', 'GRADIENT_EXPLOSION']

# Control case
NO_FAULT = 'NONE'

def main():
    """
    Main function to orchestrate and run all experiments.
    """
    print("--- Starting Full Experimental Suite ---")
    experiment_counter = 0
    all_configs = []

    # --- Generate Configs for Faults with Severity ---
    for model, fault_type, (severity, params), rep in itertools.product(
        MODELS, SEVERITY_FAULTS.keys(), SEVERITY_FAULTS.items(), REPETITIONS
    ):
        # This check ensures we correctly match fault types with their params
        if fault_type != list(params.keys())[0]:
            continue

        experiment_id = f"run_{experiment_counter:03d}_{model.lower()}_{fault_type.lower()}_{severity}_rep{rep}"
        
        config = copy.deepcopy(BASE_CONFIG)
        config['experiment_id'] = experiment_id
        config['model_type'] = model
        config['fault_injection']['type'] = fault_type
        # Update params based on severity
        config['fault_injection']['params'].update(list(params.values())[0][severity])

        all_configs.append(config)
        experiment_counter += 1

    # --- Generate Configs for Simple Faults ---
    for model, fault_type, rep in itertools.product(MODELS, SIMPLE_FAULTS, REPETITIONS):
        experiment_id = f"run_{experiment_counter:03d}_{model.lower()}_{fault_type.lower()}_rep{rep}"
        
        config = copy.deepcopy(BASE_CONFIG)
        config['experiment_id'] = experiment_id
        config['model_type'] = model
        config['fault_injection']['type'] = fault_type

        all_configs.append(config)
        experiment_counter += 1

    # --- Generate Configs for No-Fault Control Group ---
    for model, rep in itertools.product(MODELS, REPETITIONS):
        experiment_id = f"run_{experiment_counter:03d}_{model.lower()}_control_rep{rep}"

        config = copy.deepcopy(BASE_CONFIG)
        config['experiment_id'] = experiment_id
        config['model_type'] = model
        config['fault_injection']['type'] = NO_FAULT
        
        all_configs.append(config)
        experiment_counter += 1

    # --- Run All Experiments ---
    print(f"Generated a total of {len(all_configs)} experiment configurations.")
    for i, config in enumerate(all_configs):
        print(f"\n--- Running Experiment {i+1}/{len(all_configs)}: {config['experiment_id']} ---")
        try:
            simulator = TrainingSimulator(config)
            simulator.run()
        except Exception as e:
            print(f"!!! Experiment {config['experiment_id']} failed with an error: {e} !!!")

    print("\n--- Full Experimental Suite Completed ---")


if __name__ == '__main__':
    main()
