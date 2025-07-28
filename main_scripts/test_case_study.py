# test_case_study.py for GPT-2 study
"""
Test script to run a minimal case study
"""

import torch
from case_study_qwen_enhanced import ExperimentConfig, CaseStudyTrainer

def test_minimal_case_study():
    """Run a minimal test of the case study"""
    # Create test configuration
    config = ExperimentConfig(
        model_name="gpt2",  # Use small model
        dataset_name="imdb",  # Use simple dataset
        max_steps=100,  # Reduce steps for testing
        eval_every_n_steps=10,
        batch_size=1,
        fault_injection_step=50,
        use_cpu=True  # Force CPU for testing
    )
    
    print(f"Running test case study...")
    print(f"Output will be saved to: {config.output_path}")
    
    # Run experiment
    trainer = CaseStudyTrainer(config)
    trainer.train()
    trainer.save_results()
    
    # Try visualization
    try:
        from visualization import create_case_study_visualizations
        create_case_study_visualizations(config.output_path)
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"⚠️ Visualization error: {e}")
        print("✅ Training completed successfully!")
    
    return config.output_path

if __name__ == "__main__":
    output_path = test_minimal_case_study()
    print(f"\nResults saved to: {output_path}")
    print("\nTo view results:")
    print(f"  - Metrics: {output_path}/metrics.csv")
    print(f"  - Summary: {output_path}/summary.json")
    print(f"  - Plots: {output_path}/*.png")
