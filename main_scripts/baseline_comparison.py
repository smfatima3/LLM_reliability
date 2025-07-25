# baseline_comparison.py
"""
Comprehensive baseline comparison for fault detection methods
Implements multiple baseline approaches and compares with R-Metric
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class BaselineMethod:
    """Base class for baseline methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.alerts = []
        self.scores = []
        
    def update(self, step: int, metrics: Dict) -> bool:
        """Update method and return alert status"""
        raise NotImplementedError
        
    def reset(self):
        """Reset method state"""
        self.alerts = []
        self.scores = []


class SimpleHeuristicBaseline(BaselineMethod):
    """Simple heuristic based on consecutive loss increases"""
    
    def __init__(self, threshold: int = 3):
        super().__init__("Simple Heuristic")
        self.threshold = threshold
        self.consecutive_increases = 0
        self.prev_loss = None
        
    def update(self, step: int, metrics: Dict) -> bool:
        current_loss = metrics['train_loss']
        
        if self.prev_loss is not None:
            if current_loss > self.prev_loss:
                self.consecutive_increases += 1
            else:
                self.consecutive_increases = 0
        
        self.prev_loss = current_loss
        alert = self.consecutive_increases >= self.threshold
        
        self.alerts.append(alert)
        self.scores.append(self.consecutive_increases / self.threshold)
        
        return alert


class LossSpikeBaseline(BaselineMethod):
    """Detect sudden spikes in loss using z-score"""
    
    def __init__(self, window_size: int = 20, z_threshold: float = 3.0):
        super().__init__("Loss Spike Detection")
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.loss_history = []
        
    def update(self, step: int, metrics: Dict) -> bool:
        current_loss = metrics['train_loss']
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < self.window_size:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        # Keep only recent history
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size*2:]
        
        # Calculate z-score
        recent_losses = self.loss_history[-self.window_size:]
        mean_loss = np.mean(recent_losses[:-1])  # Exclude current
        std_loss = np.std(recent_losses[:-1])
        
        if std_loss > 0:
            z_score = (current_loss - mean_loss) / std_loss
            score = min(abs(z_score) / self.z_threshold, 1.0)
            alert = z_score > self.z_threshold
        else:
            z_score = 0
            score = 0
            alert = False
        
        self.alerts.append(alert)
        self.scores.append(score)
        
        return alert


class GradientNormBaseline(BaselineMethod):
    """Monitor gradient norms for instability"""
    
    def __init__(self, threshold: float = 100.0, window_size: int = 10):
        super().__init__("Gradient Norm Monitor")
        self.threshold = threshold
        self.window_size = window_size
        self.grad_history = []
        
    def update(self, step: int, metrics: Dict) -> bool:
        # Assume gradient norm is provided in metrics
        grad_norm = metrics.get('grad_norm', np.random.normal(10, 2))
        self.grad_history.append(grad_norm)
        
        if len(self.grad_history) > self.window_size:
            self.grad_history = self.grad_history[-self.window_size:]
        
        # Check both absolute threshold and relative increase
        alert = grad_norm > self.threshold
        
        if len(self.grad_history) >= 3:
            recent_mean = np.mean(self.grad_history[-3:])
            historical_mean = np.mean(self.grad_history[:-3]) if len(self.grad_history) > 3 else recent_mean
            if historical_mean > 0:
                relative_increase = recent_mean / historical_mean
                alert = alert or relative_increase > 5.0
        
        score = min(grad_norm / self.threshold, 1.0)
        
        self.alerts.append(alert)
        self.scores.append(score)
        
        return alert


class IsolationForestBaseline(BaselineMethod):
   """Isolation Forest anomaly detection"""
   
   def __init__(self, contamination: float = 0.1, window_size: int = 100):
       super().__init__("Isolation Forest")
       self.contamination = contamination
       self.window_size = window_size
       self.feature_history = []
       self.model = IsolationForest(contamination=contamination, random_state=42)
       self.scaler = StandardScaler()
       self.is_fitted = False
       
   def update(self, step: int, metrics: Dict) -> bool:
       # Extract features
       features = [
           metrics['train_loss'],
           metrics.get('grad_norm', np.random.normal(10, 2)),
           metrics.get('learning_rate', 5e-5),
           metrics.get('loss_variance', np.random.normal(0.1, 0.02))
       ]
       
       self.feature_history.append(features)
       
       if len(self.feature_history) < 50:  # Need minimum samples
           self.alerts.append(False)
           self.scores.append(0.0)
           return False
       
       # Keep window of recent data
       if len(self.feature_history) > self.window_size:
           self.feature_history = self.feature_history[-self.window_size:]
       
       # Fit and predict
       X = np.array(self.feature_history)
       X_scaled = self.scaler.fit_transform(X)
       
       self.model.fit(X_scaled[:-1])  # Fit on historical data
       prediction = self.model.predict(X_scaled[-1:])  # Predict current
       score = -self.model.score_samples(X_scaled[-1:])[0]  # Anomaly score
       
       alert = prediction[0] == -1  # -1 indicates anomaly
       normalized_score = 1 / (1 + np.exp(-score))  # Sigmoid normalization
       
       self.alerts.append(alert)
       self.scores.append(normalized_score)
       
       return alert


class LSTMBaseline(BaselineMethod):
   """LSTM-based anomaly detection"""
   
   def __init__(self, sequence_length: int = 20, threshold: float = 2.0):
       super().__init__("LSTM Anomaly Detector")
       self.sequence_length = sequence_length
       self.threshold = threshold
       self.loss_history = []
       self.model = self._build_model()
       self.is_trained = False
       
   def _build_model(self):
       """Build simple LSTM model"""
       class SimpleLSTM(nn.Module):
           def __init__(self, input_size=1, hidden_size=32, num_layers=2):
               super().__init__()
               self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
               self.fc = nn.Linear(hidden_size, 1)
               
           def forward(self, x):
               lstm_out, _ = self.lstm(x)
               return self.fc(lstm_out[:, -1, :])
       
       return SimpleLSTM()
   
   def update(self, step: int, metrics: Dict) -> bool:
       current_loss = metrics['train_loss']
       self.loss_history.append(current_loss)
       
       if len(self.loss_history) < self.sequence_length + 50:
           self.alerts.append(False)
           self.scores.append(0.0)
           return False
       
       # Create sequences
       sequences = []
       targets = []
       
       for i in range(len(self.loss_history) - self.sequence_length):
           seq = self.loss_history[i:i+self.sequence_length]
           target = self.loss_history[i+self.sequence_length]
           sequences.append(seq)
           targets.append(target)
       
       if len(sequences) < 30:  # Need minimum training data
           self.alerts.append(False)
           self.scores.append(0.0)
           return False
       
       # Convert to tensors
       X = torch.tensor(sequences[:-1], dtype=torch.float32).unsqueeze(-1)
       y = torch.tensor(targets[:-1], dtype=torch.float32).unsqueeze(-1)
       
       # Train model if needed
       if not self.is_trained or step % 100 == 0:
           self._train_model(X, y)
           self.is_trained = True
       
       # Predict current
       current_seq = torch.tensor([sequences[-1]], dtype=torch.float32).unsqueeze(-1)
       with torch.no_grad():
           prediction = self.model(current_seq).item()
       
       # Calculate error
       actual = targets[-1]
       error = abs(actual - prediction)
       relative_error = error / (abs(prediction) + 1e-6)
       
       alert = relative_error > self.threshold
       score = min(relative_error / self.threshold, 1.0)
       
       self.alerts.append(alert)
       self.scores.append(score)
       
       return alert
   
   def _train_model(self, X, y, epochs=10):
       """Quick training of LSTM"""
       optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
       criterion = nn.MSELoss()
       
       for _ in range(epochs):
           self.model.train()
           optimizer.zero_grad()
           outputs = self.model(X)
           loss = criterion(outputs, y)
           loss.backward()
           optimizer.step()


class MovingAverageBaseline(BaselineMethod):
   """Moving average with adaptive thresholds"""
   
   def __init__(self, window_size: int = 20, sensitivity: float = 2.5):
       super().__init__("Adaptive Moving Average")
       self.window_size = window_size
       self.sensitivity = sensitivity
       self.loss_history = []
       
   def update(self, step: int, metrics: Dict) -> bool:
       current_loss = metrics['train_loss']
       self.loss_history.append(current_loss)
       
       if len(self.loss_history) < self.window_size:
           self.alerts.append(False)
           self.scores.append(0.0)
           return False
       
       # Calculate moving statistics
       recent = self.loss_history[-self.window_size:]
       ma = np.mean(recent)
       std = np.std(recent)
       
       # Adaptive threshold based on recent volatility
       threshold = ma + self.sensitivity * std
       
       alert = current_loss > threshold
       score = min((current_loss - ma) / (self.sensitivity * std + 1e-6), 1.0) if std > 0 else 0.0
       score = max(0, score)  # Ensure non-negative
       
       self.alerts.append(alert)
       self.scores.append(score)
       
       return alert


class EnsembleBaseline(BaselineMethod):
   """Ensemble of multiple baselines"""
   
   def __init__(self, methods: List[BaselineMethod], voting: str = 'soft'):
       super().__init__("Ensemble Method")
       self.methods = methods
       self.voting = voting  # 'hard' or 'soft'
       
   def update(self, step: int, metrics: Dict) -> bool:
       # Update all methods
       alerts = []
       scores = []
       
       for method in self.methods:
           alert = method.update(step, metrics)
           alerts.append(alert)
           if method.scores:
               scores.append(method.scores[-1])
           else:
               scores.append(0.0)
       
       # Ensemble decision
       if self.voting == 'hard':
           # Majority voting
           alert = sum(alerts) > len(alerts) / 2
           score = sum(alerts) / len(alerts)
       else:
           # Soft voting using scores
           avg_score = np.mean(scores)
           alert = avg_score > 0.5
           score = avg_score
       
       self.alerts.append(alert)
       self.scores.append(score)
       
       return alert
   
   def reset(self):
       """Reset all methods"""
       super().reset()
       for method in self.methods:
           method.reset()


class BaselineComparison:
   """Run and compare all baseline methods"""
   
   def __init__(self, config_path: str = None):
       self.methods = self._initialize_methods()
       self.results = defaultdict(list)
       self.config = self._load_config(config_path)
       
   def _load_config(self, config_path: str) -> Dict:
       """Load configuration"""
       if config_path and Path(config_path).exists():
           with open(config_path, 'r') as f:
               return json.load(f)
       else:
           # Default configuration
           return {
               'fault_injection_step': 400,
               'max_steps': 800,
               'r_metric_alert_threshold': 0.6
           }
   
   def _initialize_methods(self) -> List[BaselineMethod]:
       """Initialize all baseline methods"""
       methods = [
           SimpleHeuristicBaseline(threshold=3),
           LossSpikeBaseline(window_size=20, z_threshold=2.5),
           GradientNormBaseline(threshold=50.0),
           IsolationForestBaseline(contamination=0.1),
           LSTMBaseline(sequence_length=20),
           MovingAverageBaseline(window_size=20, sensitivity=2.0)
       ]
       
       # Add ensemble
       ensemble_methods = [
           SimpleHeuristicBaseline(threshold=3),
           LossSpikeBaseline(window_size=20, z_threshold=2.5),
           GradientNormBaseline(threshold=50.0)
       ]
       methods.append(EnsembleBaseline(ensemble_methods, voting='soft'))
       
       return methods
   
   def run_comparison(self, metrics_df: pd.DataFrame) -> Dict:
       """Run all methods on the data"""
       # Reset all methods
       for method in self.methods:
           method.reset()
       
       # Process each step
       for idx, row in metrics_df.iterrows():
           step = row['step']
           metrics = row.to_dict()
           
           # Update each method
           for method in self.methods:
               alert = method.update(step, metrics)
       
       # Compile results
       results = {}
       fault_step = self.config['fault_injection_step']
       
       for method in self.methods:
           # Find first alert
           alerts = np.array(method.alerts)
           alert_indices = np.where(alerts)[0]
           
           if len(alert_indices) > 0:
               first_alert_idx = alert_indices[0]
               first_alert_step = metrics_df.iloc[first_alert_idx]['step']
               lead_time = fault_step - first_alert_step
           else:
               first_alert_step = None
               lead_time = None
           
           # Calculate metrics
           results[method.name] = {
               'first_alert_step': first_alert_step,
               'lead_time': lead_time,
               'total_alerts': sum(alerts),
               'alert_rate': np.mean(alerts),
               'scores': method.scores,
               'detected': first_alert_step is not None
           }
       
       return results
   
   def evaluate_methods(self, metrics_df: pd.DataFrame, r_metric_results: Dict) -> pd.DataFrame:
       """Evaluate and compare all methods"""
       results = self.run_comparison(metrics_df)
       
       # Add R-Metric results
       r_metric_alerts = metrics_df['r_metric'] > self.config['r_metric_alert_threshold']
       r_alert_indices = np.where(r_metric_alerts)[0]
       
       if len(r_alert_indices) > 0:
           r_first_alert = metrics_df.iloc[r_alert_indices[0]]['step']
           r_lead_time = self.config['fault_injection_step'] - r_first_alert
       else:
           r_first_alert = None
           r_lead_time = None
       
       results['R-Metric'] = {
           'first_alert_step': r_first_alert,
           'lead_time': r_lead_time,
           'total_alerts': sum(r_metric_alerts),
           'alert_rate': np.mean(r_metric_alerts),
           'scores': metrics_df['r_metric'].tolist(),
           'detected': r_first_alert is not None
       }
       
       # Create comparison dataframe
       comparison_data = []
       for method_name, method_results in results.items():
           comparison_data.append({
               'Method': method_name,
               'Detected': method_results['detected'],
               'First Alert': method_results['first_alert_step'],
               'Lead Time': method_results['lead_time'] if method_results['lead_time'] else -999,
               'Total Alerts': method_results['total_alerts'],
               'Alert Rate': method_results['alert_rate'],
               'Early Detection': 1 if method_results['lead_time'] and method_results['lead_time'] > 0 else 0
           })
       
       comparison_df = pd.DataFrame(comparison_data)
       comparison_df = comparison_df.sort_values('Lead Time', ascending=False)
       
       return comparison_df, results
   
   def plot_comparison(self, metrics_df: pd.DataFrame, results: Dict, output_path: Path):
       """Create comparison visualizations"""
       fig, axes = plt.subplots(3, 2, figsize=(15, 12))
       axes = axes.flatten()
       
       fault_step = self.config['fault_injection_step']
       
       # 1. Score evolution for all methods
       ax = axes[0]
       for method_name, method_results in results.items():
           if method_results['scores']:
               ax.plot(metrics_df['step'][:len(method_results['scores'])], 
                      method_results['scores'], label=method_name, alpha=0.7)
       
       ax.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
       ax.set_xlabel('Step')
       ax.set_ylabel('Anomaly Score')
       ax.set_title('Anomaly Scores Evolution')
       ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       ax.grid(True, alpha=0.3)
       
       # 2. Lead time comparison
       ax = axes[1]
       methods = []
       lead_times = []
       colors = []
       
       for method_name, method_results in results.items():
           if method_results['detected']:
               methods.append(method_name)
               lead_times.append(method_results['lead_time'])
               colors.append('green' if method_results['lead_time'] > 0 else 'red')
       
       bars = ax.barh(methods, lead_times, color=colors)
       ax.axvline(x=0, color='black', linestyle='-')
       ax.set_xlabel('Lead Time (steps)')
       ax.set_title('Detection Lead Times')
       
       # Add value labels
       for bar, lt in zip(bars, lead_times):
           ax.text(bar.get_width() + 5 if bar.get_width() > 0 else bar.get_width() - 5,
                  bar.get_y() + bar.get_height()/2, f'{int(lt)}',
                  ha='left' if bar.get_width() > 0 else 'right', va='center')
       
       # 3. Detection timeline
       ax = axes[2]
       y_pos = 0
       for method_name, method_results in results.items():
           if method_results['detected']:
               ax.scatter(method_results['first_alert_step'], y_pos, s=100, 
                         label=method_name)
               y_pos += 1
       
       ax.axvline(x=fault_step, color='red', linestyle='--', label='Fault')
       ax.set_xlabel('Step')
       ax.set_yticks([])
       ax.set_title('Alert Timeline')
       ax.legend()
       ax.grid(True, axis='x', alpha=0.3)
       
       # 4. False positive analysis
       ax = axes[3]
       pre_fault_steps = metrics_df[metrics_df['step'] < fault_step]
       
       fp_data = []
       for method_name, method_results in results.items():
           if method_results['scores']:
               pre_fault_alerts = sum(method_results['scores'][:len(pre_fault_steps)] > np.array([0.5]))
               fp_rate = pre_fault_alerts / len(pre_fault_steps) if len(pre_fault_steps) > 0 else 0
               fp_data.append({'Method': method_name, 'FP Rate': fp_rate})
       
       fp_df = pd.DataFrame(fp_data).sort_values('FP Rate')
       bars = ax.bar(range(len(fp_df)), fp_df['FP Rate'])
       ax.set_xticks(range(len(fp_df)))
       ax.set_xticklabels(fp_df['Method'], rotation=45, ha='right')
       ax.set_ylabel('False Positive Rate')
       ax.set_title('Pre-Fault False Positive Rates')
       
       # Add value labels
       for bar in bars:
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height,
                  f'{height:.3f}', ha='center', va='bottom')
       
       # 5. ROC-style analysis
       ax = axes[4]
       for method_name, method_results in results.items():
           if method_results['scores'] and len(method_results['scores']) == len(metrics_df):
               # Create binary labels (post-fault = 1, pre-fault = 0)
               labels = (metrics_df['step'] >= fault_step).astype(int)
               scores = method_results['scores']
               
               # Calculate TPR and FPR for different thresholds
               thresholds = np.linspace(0, 1, 50)
               tpr_list = []
               fpr_list = []
               
               for thresh in thresholds:
                   predictions = np.array(scores) > thresh
                   tp = sum((predictions == 1) & (labels == 1))
                   fp = sum((predictions == 1) & (labels == 0))
                   tn = sum((predictions == 0) & (labels == 0))
                   fn = sum((predictions == 0) & (labels == 1))
                   
                   tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                   fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                   
                   tpr_list.append(tpr)
                   fpr_list.append(fpr)
               
               ax.plot(fpr_list, tpr_list, label=method_name)
       
       ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
       ax.set_xlabel('False Positive Rate')
       ax.set_ylabel('True Positive Rate')
       ax.set_title('ROC-style Analysis')
       ax.legend()
       ax.grid(True, alpha=0.3)
       
       # 6. Summary statistics
       ax = axes[5]
       ax.axis('off')
       
       # Create summary table
       summary_text = "Method Performance Summary\n" + "="*40 + "\n\n"
       summary_text += f"{'Method':<20} {'Detected':<10} {'Lead Time':<12} {'FP Rate':<10}\n"
       summary_text += "-"*52 + "\n"
       
       for method_name, method_results in results.items():
           detected = "Yes" if method_results['detected'] else "No"
           lead_time = f"{method_results['lead_time']}" if method_results['lead_time'] else "N/A"
           
           # Calculate FP rate
           if method_results['scores']:
               pre_fault_scores = method_results['scores'][:len(pre_fault_steps)]
               fp_rate = sum(np.array(pre_fault_scores) > 0.5) / len(pre_fault_scores) if pre_fault_scores else 0
           else:
               fp_rate = 0
           
           summary_text += f"{method_name:<20} {detected:<10} {lead_time:<12} {fp_rate:<10.3f}\n"
       
       ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
       
       # Save figure
       plt.suptitle('Baseline Methods Comparison', fontsize=16, fontweight='bold')
       plt.tight_layout()
       plt.savefig(output_path / 'baseline_comparison_detailed.png', dpi=300, bbox_inches='tight')
       plt.savefig(output_path / 'baseline_comparison_detailed.pdf', bbox_inches='tight')
       plt.close()


def run_baseline_comparison(metrics_path: str, output_path: str):
   """Main function to run baseline comparison"""
   # Load data
   metrics_df = pd.read_csv(metrics_path)
   output_dir = Path(output_path)
   
   # Initialize comparison
   comparison = BaselineComparison()
   
   # Run evaluation
   comparison_df, detailed_results = comparison.evaluate_methods(metrics_df, {})
   
   # Save results
   comparison_df.to_csv(output_dir / 'baseline_comparison_results.csv', index=False)
   
   # Create visualizations
   comparison.plot_comparison(metrics_df, detailed_results, output_dir)
   
   # Print summary
   print("\nBaseline Comparison Results:")
   print("="*60)
   print(comparison_df.to_string(index=False))
   
   return comparison_df, detailed_results


if __name__ == "__main__":
   import sys
   if len(sys.argv) > 2:
       metrics_path = sys.argv[1]
       output_path = sys.argv[2]
       run_baseline_comparison(metrics_path, output_path)
   else:
       print("Usage: python baseline_comparison.py <metrics_csv_path> <output_directory>")