import itertools
import copy
import os
from simulator import TrainingSimulator, CONFIG as BASE_CONFIG

def main():
    MODELS = ['GPT-2', 'T5', 'BERT']
    REPETITIONS = [1, 2, 3, 4, 5, 6]
    FAULT_TYPES = ['NODE_FAILURE', 'GRADIENT_EXPLOSION', 'LR_SPIKE', 'DATA_CORRUPTION', 'NONE']
    ABLATION_MODES = ['FULL_METRIC', 'LAMBDA_ONLY', 'SIGMA_ONLY', 'DELTA_L_ONLY']

    for mode in ABLATION_MODES:
        print("\n" + "="*20 + f" RUNNING ABLATION MODE: {mode} " + "="*20)
        experiment_counter = 0
        
        for model, fault_type, rep in itertools.product(MODELS, FAULT_TYPES, REPETITIONS):
            config = copy.deepcopy(BASE_CONFIG)
            config['METRIC_MODE'] = mode
            config['experiment_id'] = f"run_{experiment_counter:03d}_{model.lower()}_{fault_type.lower()}_rep{rep}"
            config['model_type'] = model
            config['fault_injection']['type'] = fault_type
            
            print(f"\n--- Running Experiment: {config['experiment_id']} ---")
            try:
                simulator = TrainingSimulator(config)
                simulator.run()
            except Exception as e:
                print(f"!!! Experiment {config['experiment_id']} failed with an error: {e} !!!")
            
            experiment_counter += 1

    print("\n--- All Ablation Studies Completed ---")

if __name__ == '__main__':
    main()
