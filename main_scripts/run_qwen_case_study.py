# run_qwen_case_study.py
"""
Simplified script to run Qwen case study with optimal settings
"""

import os
import torch
from case_study_qwen import ExperimentConfig, CaseStudyTrainer

def run_qwen_case_study(model_size="auto", custom_config=None):
    """
    Run Qwen case study with automatic configuration
    
    Args:
        model_size: "auto", "0.5B", "1.5B", "7B"
        custom_config: Optional custom configuration dictionary
    """
    # Auto-detect best model size based on available resources
    if model_size == "auto":
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 8:
                model_size = "0.5B"
            elif gpu_memory < 16:
                model_size = "1.5B"
            else:
                model_size = "7B"
        else:
            model_size = "0.5B"  # CPU mode
    
    # Model configurations
    model_configs = {
        "0.5B": {
            "model_name": "Qwen/Qwen2-0.5B-Instruct",
            "batch_size": 4,
            "gradient_accumulation_steps": 1,
            "use_8bit": False,
            "max_seq_length": 256
        },
        "1.5B": {
            "model_name": "Qwen/Qwen2-1.5B-Instruct",
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "use_8bit": True,
            "max_seq_length": 384
        },
        "7B": {
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "use_8bit": True,
            "max_seq_length": 512
        }
    }
    
    # Get configuration
    base_config = model_configs[model_size]
    
    # Apply custom configuration if provided
    if custom_config:
        base_config.update(custom_config)
    
    # Create experiment configuration
    config = ExperimentConfig(**base_config)
    
    print(f"\n🚀 Running Qwen {model_size} Case Study")
    print(f"Model: {config.model_name}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    # Run experiment
    trainer = CaseStudyTrainer(config)
    trainer.train()
    trainer.save_results()
    
    # Try to generate visualizations
    try:
        from visualization import create_case_study_visualizations
        create_case_study_visualizations(config.output_path)
        print("\n✅ Visualizations created successfully!")
    except Exception as e:
        print(f"\n⚠️ Could not create visualizations: {e}")
    
    print(f"\n✅ Case study completed!")
    print(f"Results saved to: {config.output_path}")
    
    return config.output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Qwen case study")
    parser.add_argument("--model-size", type=str, default="auto",
                        choices=["auto", "0.5B", "1.5B", "7B"],
                        help="Model size to use")
    parser.add_argument("--max-steps", type=int, default=800,
                        help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides default)")
    parser.add_argument("--no-8bit", action="store_true",
                        help="Disable 8-bit quantization")
    
    args = parser.parse_args()
    
    # Build custom config
    custom_config = {
        "max_steps": args.max_steps
    }
    
    if args.batch_size:
        custom_config["batch_size"] = args.batch_size
    
    if args.no_8bit:
        custom_config["use_8bit"] = False
    
    # Set HF token if available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        custom_config["hf_token"] = hf_token
    
    # Run case study
    run_qwen_case_study(args.model_size, custom_config)