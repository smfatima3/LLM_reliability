# case_study_qwen_enhanced.py
"""
Enhanced Case Study: Proactive Failure Detection in Qwen 2.5B Training
Implements the R-Metric monitoring system with comprehensive baselines
"""

import os
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    logging as transformers_logging
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
transformers_logging.set_verbosity_error()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for the case study experiment"""
    # Model configuration - using Qwen model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # Updated to use Qwen2.5
    dataset_name: str = "HuggingFaceH4/ultrachat_200k"  # Better dataset for instruction tuning
    dataset_config: str = None
    dataset_split: str = "train_sft"  # Specific split for SFT
    
    # Training configuration
    max_steps: int = 800
    eval_every_n_steps: int = 20
    batch_size: int = 1  # Reduced for 7B model
    learning_rate: float = 5e-5
    warmup_steps: int = 50
    max_seq_length: int = 512  # Qwen supports longer context
    gradient_accumulation_steps: int = 4  # To simulate larger batch
    
    # Fault injection configuration
    fault_injection_step: int = 400
    fault_type: str = "LR_SPIKE"
    lr_spike_factor: float = 15.0
    lr_spike_duration: int = 20
    
    # Monitoring configuration with weights from test results
    r_metric_alert_threshold: float = 0.60
    r_metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda": 0.0,     # From test results
        "sigma_sq": 0.52,  # From test results
        "delta_l": 0.78    # From test results - primary contributor
    })
    
    # Window sizes
    loss_history_window: int = 10
    gradient_history_window: int = 20
    hardware_event_window: int = 50
    
    # Output configuration
    output_dir: str = "case_study_results"
    experiment_name: str = field(default_factory=lambda: f"qwen_case_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Device configuration
    device: str = "auto"
    mixed_precision: bool = True
    use_8bit: bool = True  # For memory efficiency with 7B model
    use_cpu: bool = False
    
    # HuggingFace token
    hf_token: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization setup"""
        if self.device == "auto":
            if self.use_cpu:
                self.device = "cpu"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create output directory
        self.output_path = Path(self.output_dir) / self.experiment_name
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Get HF token from environment if not provided
        if self.hf_token is None:
            self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary with JSON-serializable values"""
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['output_path'] = str(self.output_path)
        # Don't save token in logs
        config_dict.pop('hf_token', None)
        return config_dict


class MetricTracker:
    """Base class for tracking metrics with history"""
    
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.all_values = []
    
    def update(self, value: float):
        """Update metric with new value"""
        if not np.isnan(value) and not np.isinf(value):
            self.history.append(value)
            self.all_values.append(value)
    
    def get_mean(self) -> float:
        """Get mean of recent values"""
        return np.mean(list(self.history)) if self.history else 0.0
    
    def get_std(self) -> float:
        """Get standard deviation of recent values"""
        return np.std(list(self.history)) if len(self.history) > 1 else 0.0
    
    def normalize(self, value: float) -> float:
        """Normalize value based on history"""
        if len(self.history) < 2:
            return 0.0
        
        min_val = min(self.history)
        max_val = max(self.history)
        
        if max_val == min_val:
            return 0.0
        
        return (value - min_val) / (max_val - min_val)


class ReliabilityMonitor:
    """Enhanced reliability monitoring system with R-Metric"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize metric trackers
        self.loss_tracker = MetricTracker(config.loss_history_window)
        self.gradient_tracker = MetricTracker(config.gradient_history_window)
        self.hardware_tracker = MetricTracker(config.hardware_event_window)
        
        # Initialize component histories for normalization
        self.lambda_history = deque(maxlen=100)
        self.sigma_sq_history = deque(maxlen=100)
        self.delta_l_history = deque(maxlen=100)
        
        # Metrics storage
        self.metrics_history = []
        
    def simulate_hardware_events(self) -> float:
        """Simulate hardware failure rate (λ)"""
        # In real deployment, this would come from actual hardware monitoring
        # For simulation, we use a low base rate with occasional spikes
        base_rate = 0.1
        if np.random.random() < 0.05:  # 5% chance of hardware event
            return base_rate + np.random.exponential(0.5)
        return base_rate
    
    def calculate_gradient_variance(self, gradients: List[torch.Tensor]) -> float:
        """Calculate gradient variance across parameters (σ²)"""
        if not gradients:
            return 0.0
        
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                grad_norm = grad.norm().item()
                if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                    grad_norms.append(grad_norm)
        
        if not grad_norms:
            return 0.0
        
        return np.var(grad_norms)
    
    def calculate_loss_drift(self, current_loss: float) -> float:
        """Calculate validation loss drift (ΔL)"""
        if np.isnan(current_loss) or np.isinf(current_loss):
            return 10.0  # High value for unstable loss
        
        if len(self.loss_tracker.history) < 2:
            self.loss_tracker.update(current_loss)
            return 0.0
        
        mean_loss = self.loss_tracker.get_mean()
        drift = abs(current_loss - mean_loss)
        self.loss_tracker.update(current_loss)
        
        return drift
    
    def calculate_r_metric(self, 
                          current_loss: float,
                          gradients: Optional[List[torch.Tensor]] = None) -> Dict[str, float]:
        """Calculate the composite R-Metric"""
        # Calculate components
        lambda_val = self.simulate_hardware_events()
        sigma_sq_val = self.calculate_gradient_variance(gradients) if gradients else 0.0
        delta_l_val = self.calculate_loss_drift(current_loss)
        
        # Update histories
        self.lambda_history.append(lambda_val)
        self.sigma_sq_history.append(sigma_sq_val)
        self.delta_l_history.append(delta_l_val)
        
        # Normalize components
        lambda_norm = self._normalize_value(lambda_val, self.lambda_history)
        sigma_sq_norm = self._normalize_value(sigma_sq_val, self.sigma_sq_history)
        delta_l_norm = self._normalize_value(delta_l_val, self.delta_l_history)
        
        # Calculate weighted R-Metric
        weights = self.config.r_metric_weights
        r_metric = (
            weights["lambda"] * lambda_norm +
            weights["sigma_sq"] * sigma_sq_norm +
            weights["delta_l"] * delta_l_norm
        )
        
        return {
            "r_metric": r_metric,
            "lambda": lambda_val,
            "lambda_norm": lambda_norm,
            "sigma_sq": sigma_sq_val,
            "sigma_sq_norm": sigma_sq_norm,
            "delta_l": delta_l_val,
            "delta_l_norm": delta_l_norm
        }
    
    def _normalize_value(self, value: float, history: deque) -> float:
        """Normalize value based on history"""
        if len(history) < 2:
            return 0.0
        
        min_val = min(history)
        max_val = max(history)
        
        if max_val == min_val:
            return 0.0
        
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)


class BaselineMonitors:
    """Collection of baseline monitoring approaches"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Simple heuristics
        self.consecutive_loss_increases = 0
        self.loss_spike_threshold = 3.0
        
        # Isolation Forest for anomaly detection
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.loss_history_for_if = []
        
        # Gradient norm monitoring
        self.grad_norm_threshold = 100.0
        self.grad_norm_history = deque(maxlen=20)
        
    def update_simple_heuristic(self, current_loss: float, prev_loss: float) -> bool:
        """Simple heuristic: consecutive loss increases"""
        if current_loss > prev_loss:
            self.consecutive_loss_increases += 1
        else:
            self.consecutive_loss_increases = 0
        
        return self.consecutive_loss_increases >= 3
    
    def update_loss_spike(self, current_loss: float, loss_history: List[float]) -> bool:
        """Detect sudden loss spikes"""
        if len(loss_history) < 10:
            return False
        
        mean_loss = np.mean(loss_history[-10:])
        std_loss = np.std(loss_history[-10:])
        
        if std_loss > 0:
            z_score = (current_loss - mean_loss) / std_loss
            return z_score > self.loss_spike_threshold
        
        return False
    
    def update_isolation_forest(self, current_loss: float) -> bool:
        """Isolation Forest anomaly detection"""
        self.loss_history_for_if.append([current_loss])
        
        if len(self.loss_history_for_if) < 50:
            return False
        
        # Fit on historical data
        X = np.array(self.loss_history_for_if[-100:])
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        
        # Predict on current value
        current_scaled = self.scaler.transform([[current_loss]])
        prediction = self.isolation_forest.predict(current_scaled)
        
        return prediction[0] == -1  # -1 indicates anomaly
    
    def update_gradient_monitoring(self, gradients: List[torch.Tensor]) -> bool:
        """Monitor gradient norms for explosion"""
        if not gradients:
            return False
        
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                grad_norm = grad.norm().item()
                if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                    grad_norms.append(grad_norm)
        
        if not grad_norms:
            return False
        
        max_grad_norm = max(grad_norms)
        self.grad_norm_history.append(max_grad_norm)
        
        return max_grad_norm > self.grad_norm_threshold


class FaultInjector:
    """Manages fault injection for experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fault_active = False
        self.fault_start_step = None
        self.original_lr = None
        
    def should_inject(self, step: int) -> bool:
        """Check if fault should be injected at current step"""
        return step == self.config.fault_injection_step
    
    def inject_fault(self, optimizer: torch.optim.Optimizer, step: int):
        """Inject fault into training process"""
        if self.config.fault_type == "LR_SPIKE":
            self._inject_lr_spike(optimizer, step)
        elif self.config.fault_type == "GRADIENT_NOISE":
            # Could implement other fault types
            pass
        
    def _inject_lr_spike(self, optimizer: torch.optim.Optimizer, step: int):
        """Inject learning rate spike"""
        if not self.fault_active:
            self.fault_active = True
            self.fault_start_step = step
            self.original_lr = optimizer.param_groups[0]['lr']
            
            new_lr = self.original_lr * self.config.lr_spike_factor
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            logger.warning(f"FAULT INJECTED: LR spiked from {self.original_lr:.2e} to {new_lr:.2e}")
    
    def should_recover(self, step: int) -> bool:
        """Check if fault should be recovered"""
        if self.fault_active and self.fault_start_step is not None:
            return step >= self.fault_start_step + self.config.lr_spike_duration
        return False
    
    def recover_fault(self, optimizer: torch.optim.Optimizer):
        """Recover from injected fault"""
        if self.fault_active and self.original_lr is not None:
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.original_lr
            
            logger.info(f"FAULT RECOVERED: LR restored to {self.original_lr:.2e}")
            self.fault_active = False


class CaseStudyTrainer:
    """Main trainer class for the case study"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.setup_model_and_data()
        self.setup_monitoring()
        
        # Results tracking
        self.results = {
            "config": config.to_dict(),
            "metrics": [],
            "alerts": {
                "r_metric": None,
                "simple_heuristic": None,
                "loss_spike": None,
                "isolation_forest": None,
                "gradient_monitoring": None
            }
        }
        
    def setup_logging(self):
        """Setup experiment logging"""
        log_file = self.config.output_path / "experiment.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        logger.info(f"Configuration: {json.dumps(self.config.to_dict(), indent=2)}")
    
    def setup_model_and_data(self):
        """Initialize model, tokenizer, and dataset"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            token=self.config.hf_token
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure quantization for memory efficiency
        quantization_config = None
        if self.config.use_8bit and self.config.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "token": self.config.hf_token,
            "use_cache": False,
            "torch_dtype": torch.float16 if self.config.mixed_precision and self.config.device == "cuda" else torch.float32
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if self.config.device == "cuda" and not quantization_config:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Move to device if not using device_map
        if self.config.device == "cpu" or quantization_config is None:
            self.model = self.model.to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        # Prepare for training if using 8bit
        if self.config.use_8bit and self.config.device == "cuda":
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Load and prepare dataset
        logger.info(f"Loading dataset: {self.config.dataset_name}")
        
        # Load dataset with proper configuration
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            token=self.config.hf_token
        )
        
        # Take a subset for experimentation
        dataset = dataset.shuffle(seed=42).select(range(min(4000, len(dataset))))
        
        # Tokenize dataset for Qwen chat format
        def format_chat_template(examples):
            """Format examples using Qwen chat template"""
            formatted_texts = []
            
            for i in range(len(examples['messages'])):
                messages = examples['messages'][i]
                
                # Apply chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
                formatted_texts.append(text)
            
            # Tokenize
            model_inputs = self.tokenizer(
                formatted_texts,
                max_length=self.config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors=None
            )
            
            # Set labels for causal language modeling
            model_inputs["labels"] = model_inputs["input_ids"].copy()
            
            return model_inputs
        
        # Apply formatting and tokenization
        self.tokenized_dataset = dataset.map(
            format_chat_template,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Formatting and tokenizing dataset"
        )
        
        self.tokenized_dataset.set_format(type='torch')
        
        # Create dataloader
        self.train_dataloader = DataLoader(
            self.tokenized_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.max_steps
        )
        
        logger.info(f"Model loaded successfully. Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def setup_monitoring(self):
        """Initialize monitoring systems"""
        self.reliability_monitor = ReliabilityMonitor(self.config)
        self.baseline_monitors = BaselineMonitors(self.config)
        self.fault_injector = FaultInjector(self.config)
        
    def get_gradients(self) -> List[torch.Tensor]:
        """Extract gradients from model parameters"""
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.clone())
        return gradients
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, List[torch.Tensor]]:
        """Perform a single training step"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Get gradients before optimizer step (only on accumulation boundaries)
        if (self.current_step + 1) % self.config.gradient_accumulation_steps == 0:
            gradients = self.get_gradients()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        else:
            gradients = []
        
        return loss.item() * self.config.gradient_accumulation_steps, gradients
    
    def evaluate_step(self, step: int, train_loss: float, gradients: List[torch.Tensor]):
        """Evaluate metrics and check alerts"""
        # Calculate R-Metric
        r_metric_results = self.reliability_monitor.calculate_r_metric(
            train_loss, gradients
        )
        
        # Calculate gradient norm for baseline
        grad_norms = []
        for grad in gradients:
            if grad is not None:
                grad_norm = grad.norm().item()
                if not np.isnan(grad_norm) and not np.isinf(grad_norm):
                    grad_norms.append(grad_norm)
        
        grad_norm = max(grad_norms) if grad_norms else 0.0
        
        # Update baseline monitors
        if len(self.results["metrics"]) > 0:
            prev_loss = self.results["metrics"][-1]["train_loss"]
            loss_history = [m["train_loss"] for m in self.results["metrics"]]
            
            heuristic_alert = self.baseline_monitors.update_simple_heuristic(
                train_loss, prev_loss
            )
            spike_alert = self.baseline_monitors.update_loss_spike(
                train_loss, loss_history
            )
            if_alert = self.baseline_monitors.update_isolation_forest(train_loss)
        else:
            heuristic_alert = spike_alert = if_alert = False
        
        grad_alert = self.baseline_monitors.update_gradient_monitoring(gradients)
        
        # Check for R-Metric alert
        r_metric_alert = r_metric_results["r_metric"] > self.config.r_metric_alert_threshold
        
        # Record alerts
        if r_metric_alert and self.results["alerts"]["r_metric"] is None:
            self.results["alerts"]["r_metric"] = step
            logger.warning(f"R-METRIC ALERT at step {step}: {r_metric_results['r_metric']:.3f}")
        
        if heuristic_alert and self.results["alerts"]["simple_heuristic"] is None:
            self.results["alerts"]["simple_heuristic"] = step
            logger.warning(f"HEURISTIC ALERT at step {step}")
        
        if spike_alert and self.results["alerts"]["loss_spike"] is None:
            self.results["alerts"]["loss_spike"] = step
            logger.warning(f"LOSS SPIKE ALERT at step {step}")
        
        if if_alert and self.results["alerts"]["isolation_forest"] is None:
            self.results["alerts"]["isolation_forest"] = step
            logger.warning(f"ISOLATION FOREST ALERT at step {step}")
        
        if grad_alert and self.results["alerts"]["gradient_monitoring"] is None:
            self.results["alerts"]["gradient_monitoring"] = step
            logger.warning(f"GRADIENT ALERT at step {step}")
        
        # Store metrics
        metrics = {
            "step": step,
            "train_loss": train_loss,
            "grad_norm": grad_norm,
            **r_metric_results,
            "heuristic_alert": heuristic_alert,
            "spike_alert": spike_alert,
            "if_alert": if_alert,
            "grad_alert": grad_alert
        }
        
        self.results["metrics"].append(metrics)
        
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        
        self.model.train()
        data_iterator = iter(self.train_dataloader)
        
        progress_bar = tqdm(range(self.config.max_steps), desc="Training")
        
        self.current_step = 0
        accumulated_loss = 0.0
        
        for step in range(self.config.max_steps):
            self.current_step = step
            
            # Get next batch
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.train_dataloader)
                batch = next(data_iterator)
            
            # Check fault injection
            if self.fault_injector.should_inject(step):
                self.fault_injector.inject_fault(self.optimizer, step)
            
            # Check fault recovery
            if self.fault_injector.should_recover(step):
                self.fault_injector.recover_fault(self.optimizer)
            
            # Training step
            try:
                loss, gradients = self.train_step(batch)
                accumulated_loss += loss
                
                # Update progress bar
                progress_bar.update(1)
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                    progress_bar.set_description(f"Step {step} | Loss: {avg_loss:.3f}")
                    accumulated_loss = 0.0
                
                # Evaluate at intervals
                if step % self.config.eval_every_n_steps == 0 and step > 0:
                    self.evaluate_step(step, loss, gradients)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"OOM at step {step}. Skipping batch.")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    progress_bar.update(1)
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error at step {step}: {e}")
                progress_bar.update(1)
        
        progress_bar.close()
        logger.info("Training completed!")
        
    def save_results(self):
        """Save experiment results"""
        # Save metrics to CSV
        df = pd.DataFrame(self.results["metrics"])
        csv_path = self.config.output_path / "metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Metrics saved to {csv_path}")
        
        # Save full results to JSON
        json_path = self.config.output_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {json_path}")
        
        # Generate summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate experiment summary"""
        summary = {
            "experiment_name": self.config.experiment_name,
            "model": self.config.model_name,
            "total_steps": self.config.max_steps,
            "fault_type": self.config.fault_type,
            "fault_step": self.config.fault_injection_step,
            "alerts": self.results["alerts"],
            "alert_lead_times": {}
        }
        
        # Calculate lead times
        for method, alert_step in self.results["alerts"].items():
            if alert_step is not None:
                lead_time = self.config.fault_injection_step - alert_step
                summary["alert_lead_times"][method] = lead_time
        
        # Save summary
        summary_path = self.config.output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            logger.info("Summary:")
       logger.info(json.dumps(summary, indent=2))


def main():
   """Main execution function"""
   # Check for HuggingFace token
   hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
   if not hf_token:
       print("⚠️  Warning: No HuggingFace token found. Some models may not be accessible.")
       print("Set your token using: export HF_TOKEN=your_token_here")
   
   # Create configuration
   config = ExperimentConfig(hf_token=hf_token)
   
   # Print configuration
   print("=" * 60)
   print("Qwen Case Study Configuration")
   print("=" * 60)
   print(f"Model: {config.model_name}")
   print(f"Dataset: {config.dataset_name}")
   print(f"Device: {config.device}")
   print(f"Mixed Precision: {config.mixed_precision}")
   print(f"8-bit Quantization: {config.use_8bit}")
   print(f"Batch Size: {config.batch_size}")
   print(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
   print(f"Effective Batch Size: {config.batch_size * config.gradient_accumulation_steps}")
   print(f"Max Steps: {config.max_steps}")
   print(f"Fault Injection Step: {config.fault_injection_step}")
   print(f"Output Path: {config.output_path}")
   print("=" * 60)
   
   # Check if CUDA is available for 7B model
   if config.device == "cpu":
       print("\n⚠️  Warning: Running on CPU. This will be very slow for a 7B model.")
       print("Consider using a GPU or the smaller Qwen2-1.5B-Instruct model.")
       response = input("Continue anyway? (y/n): ")
       if response.lower() != 'y':
           print("Exiting...")
           return
   
   # Run experiment
   trainer = CaseStudyTrainer(config)
   trainer.train()
   trainer.save_results()
   
   # Generate visualizations
   try:
       from visualization import create_case_study_visualizations
       create_case_study_visualizations(config.output_path)
       print(f"\n✅ Visualizations created successfully!")
       print(f"Results saved to: {config.output_path}")
   except FileNotFoundError as e:
       print(f"\n❌ Error: Required files not found for visualization: {e}")
       print(f"Results saved to: {config.output_path}")
   except ImportError:
       print("\n⚠️ Visualization module not found. Skipping visualizations.")
       print(f"Results saved to: {config.output_path}")
   except Exception as e:
       print(f"\n⚠️ Error creating visualizations: {e}")
       print(f"Results saved to: {config.output_path}")
   
   # Print summary
   print("\n" + "=" * 60)
   print("Experiment Summary")
   print("=" * 60)
   
   summary_path = config.output_path / "summary.json"
   if summary_path.exists():
       with open(summary_path, 'r') as f:
           summary = json.load(f)
       
       print(f"Model: {summary['model']}")
       print(f"Total Steps: {summary['total_steps']}")
       print(f"Fault Type: {summary['fault_type']}")
       print(f"Fault Step: {summary['fault_step']}")
       print("\nDetection Results:")
       print("-" * 40)
       
       for method, alert_step in summary['alerts'].items():
           method_name = method.replace('_', ' ').title()
           if alert_step is not None:
               lead_time = summary['alert_lead_times'].get(method, 'N/A')
               print(f"✓ {method_name}: Detected at step {alert_step} (Lead time: {lead_time} steps)")
           else:
               print(f"✗ {method_name}: Not detected")
   
   print("\n✅ Case study completed successfully!")


# Additional utility function for memory-constrained environments
def create_minimal_config(model_size="1.5B"):
   """Create a minimal configuration for testing"""
   if model_size == "1.5B":
       model_name = "Qwen/Qwen2-1.5B-Instruct"
       batch_size = 2
       grad_accum = 2
   elif model_size == "0.5B":
       model_name = "Qwen/Qwen2-0.5B-Instruct"
       batch_size = 4
       grad_accum = 1
   else:  # 7B
       model_name = "Qwen/Qwen2.5-7B-Instruct"
       batch_size = 1
       grad_accum = 4
   
   return ExperimentConfig(
       model_name=model_name,
       batch_size=batch_size,
       gradient_accumulation_steps=grad_accum,
       max_steps=200,  # Shorter for testing
       eval_every_n_steps=10,
       fault_injection_step=100,
       max_seq_length=256,  # Shorter sequences
       use_8bit=True  # Always use 8-bit for larger models
   )


if __name__ == "__main__":
   # Check available memory
   if torch.cuda.is_available():
       gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
       print(f"GPU Memory Available: {gpu_memory:.1f} GB")
       
       if gpu_memory < 16:
           print("\n⚠️  Limited GPU memory detected. Recommendations:")
           print("- For <8GB: Use Qwen2-0.5B-Instruct")
           print("- For 8-16GB: Use Qwen2-1.5B-Instruct with 8-bit quantization")
           print("- For >16GB: Can use Qwen2.5-7B-Instruct with 8-bit quantization")
           
           use_minimal = input("\nUse minimal configuration? (y/n): ")
           if use_minimal.lower() == 'y':
               if gpu_memory < 8:
                   config = create_minimal_config("0.5B")
               else:
                   config = create_minimal_config("1.5B")
               
               trainer = CaseStudyTrainer(config)
               trainer.train()
               trainer.save_results()
               
               print(f"\n✅ Minimal case study completed!")
               print(f"Results saved to: {config.output_path}")
           else:
               main()
       else:
           main()
   else:
       print("No GPU detected. Running on CPU...")
       main()