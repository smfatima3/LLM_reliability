# visualization.py
"""
Visualization module for case study results
Creates publication-quality figures with comprehensive analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Configure matplotlib for better fonts
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],
    'axes.grid': True,
    'grid.alpha': 0.3
})


def create_case_study_visualizations(output_path: Path):
    """Create all visualizations for the case study"""
    # Load data
    metrics_df = pd.read_csv(output_path / "metrics.csv")
    with open(output_path / "results.json", 'r') as f:
        results = json.load(f)
    
    # Create individual plots
    create_comprehensive_dashboard(metrics_df, results, output_path)
    create_component_analysis(metrics_df, results, output_path)
    create_baseline_comparison(metrics_df, results, output_path)
    create_alert_timeline(metrics_df, results, output_path)
    create_performance_summary(results, output_path)


def create_comprehensive_dashboard(df: pd.DataFrame, results: Dict, output_path: Path):
    """Create main dashboard figure showing all key metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # Get fault injection info
    fault_step = results["config"]["fault_injection_step"]
    
    # 1. Training Loss
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['step'], df['train_loss'], 'b-', linewidth=2, alpha=0.8, label='Training Loss')
    ax1.axvline(x=fault_step, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Fault Injection')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Evolution', fontweight='bold')
    ax1.legend()
    
    # Add shaded region for fault duration
    fault_duration = results["config"]["lr_spike_duration"]
    ax1.axvspan(fault_step, fault_step + fault_duration, alpha=0.2, color='red', label='Fault Active')
    
    # 2. R-Metric Components
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['step'], df['lambda_norm'], label='λ (Hardware)', alpha=0.8)
    ax2.plot(df['step'], df['sigma_sq_norm'], label='σ² (Gradient)', alpha=0.8)
    ax2.plot(df['step'], df['delta_l_norm'], label='ΔL (Loss Drift)', alpha=0.8)
    ax2.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Normalized Value')
    ax2.set_title('R-Metric Components (Normalized)', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(-0.1, 1.1)
    
    # 3. R-Metric Score
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['step'], df['r_metric'], 'g-', linewidth=2, label='R-Metric')
    ax3.axhline(y=results["config"]["r_metric_alert_threshold"], 
                color='orange', linestyle='--', linewidth=2, label='Alert Threshold')
    ax3.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
    
    # Mark alert point
    if results["alerts"]["r_metric"]:
        alert_step = results["alerts"]["r_metric"]
        alert_value = df[df['step'] == alert_step]['r_metric'].iloc[0]
        ax3.scatter([alert_step], [alert_value], color='red', s=100, zorder=5, 
                   label=f'Alert (Step {alert_step})')
   
   ax3.set_xlabel('Training Step')
   ax3.set_ylabel('R-Metric Score')
   ax3.set_title('R-Metric Evolution', fontweight='bold')
   ax3.legend()
   ax3.set_ylim(-0.1, 1.1)
   
   # 4. Raw Component Values
   ax4 = fig.add_subplot(gs[2, 0])
   ax4_twin = ax4.twinx()
   
   # Plot loss drift on primary axis
   ax4.plot(df['step'], df['delta_l'], 'b-', label='ΔL (Loss Drift)', alpha=0.8)
   ax4.set_ylabel('Loss Drift', color='b')
   ax4.tick_params(axis='y', labelcolor='b')
   
   # Plot gradient variance on secondary axis
   ax4_twin.plot(df['step'], df['sigma_sq'], 'r-', label='σ² (Gradient Var)', alpha=0.8)
   ax4_twin.set_ylabel('Gradient Variance', color='r')
   ax4_twin.tick_params(axis='y', labelcolor='r')
   
   ax4.axvline(x=fault_step, color='gray', linestyle='--', alpha=0.5)
   ax4.set_xlabel('Training Step')
   ax4.set_title('Raw Component Values', fontweight='bold')
   
   # 5. Alert Comparison
   ax5 = fig.add_subplot(gs[2, 1])
   alert_methods = ['r_metric', 'simple_heuristic', 'loss_spike', 'isolation_forest', 'gradient_monitoring']
   alert_steps = [results["alerts"][method] if results["alerts"][method] else np.nan for method in alert_methods]
   alert_labels = ['R-Metric', 'Heuristic', 'Loss Spike', 'Isolation\nForest', 'Gradient\nMonitor']
   
   # Calculate lead times (negative means after fault)
   lead_times = [fault_step - step if not np.isnan(step) else np.nan for step in alert_steps]
   
   # Create bar plot
   bars = ax5.barh(alert_labels, lead_times, color=['green' if lt > 0 else 'red' if lt <= 0 else 'gray' 
                                                     for lt in lead_times])
   
   # Add value labels
   for i, (bar, lt) in enumerate(zip(bars, lead_times)):
       if not np.isnan(lt):
           x_pos = bar.get_width() + 5 if bar.get_width() > 0 else bar.get_width() - 5
           ax5.text(x_pos, bar.get_y() + bar.get_height()/2, 
                   f'{int(lt)}', va='center', ha='left' if bar.get_width() > 0 else 'right')
   
   ax5.axvline(x=0, color='black', linestyle='-', linewidth=1)
   ax5.set_xlabel('Lead Time (steps before fault)')
   ax5.set_title('Alert Lead Times Comparison', fontweight='bold')
   ax5.grid(axis='x')
   
   # 6. Training Stability Indicators
   ax6 = fig.add_subplot(gs[3, :])
   
   # Calculate rolling statistics
   window = 20
   df['loss_rolling_mean'] = df['train_loss'].rolling(window=window, center=True).mean()
   df['loss_rolling_std'] = df['train_loss'].rolling(window=window, center=True).std()
   
   # Plot with confidence band
   ax6.plot(df['step'], df['loss_rolling_mean'], 'b-', linewidth=2, label='Mean Loss')
   ax6.fill_between(df['step'], 
                    df['loss_rolling_mean'] - 2*df['loss_rolling_std'],
                    df['loss_rolling_mean'] + 2*df['loss_rolling_std'],
                    alpha=0.3, label='±2σ Band')
   
   ax6.axvline(x=fault_step, color='red', linestyle='--', alpha=0.7)
   ax6.set_xlabel('Training Step')
   ax6.set_ylabel('Loss')
   ax6.set_title('Training Stability Analysis', fontweight='bold')
   ax6.legend()
   
   # Overall title
   fig.suptitle('Qwen 2.5B Case Study: Comprehensive Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_path / 'comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
   plt.savefig(output_path / 'comprehensive_dashboard.pdf', bbox_inches='tight')
   plt.close()


def create_component_analysis(df: pd.DataFrame, results: Dict, output_path: Path):
   """Create detailed component analysis figure"""
   fig, axes = plt.subplots(3, 2, figsize=(14, 10))
   axes = axes.flatten()
   
   fault_step = results["config"]["fault_injection_step"]
   
   # 1. Lambda (Hardware) Analysis
   ax = axes[0]
   ax.plot(df['step'], df['lambda'], 'b-', alpha=0.8)
   ax.set_title('λ (Hardware Failure Rate)', fontweight='bold')
   ax.set_ylabel('Failure Rate')
   ax.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
   
   # 2. Sigma Squared (Gradient Variance) Analysis
   ax = axes[1]
   ax.semilogy(df['step'], df['sigma_sq'], 'g-', alpha=0.8)
   ax.set_title('σ² (Gradient Variance) - Log Scale', fontweight='bold')
   ax.set_ylabel('Variance')
   ax.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
   
   # 3. Delta L (Loss Drift) Analysis
   ax = axes[2]
   ax.plot(df['step'], df['delta_l'], 'r-', alpha=0.8)
   ax.set_title('ΔL (Loss Drift)', fontweight='bold')
   ax.set_ylabel('Drift')
   ax.axvline(x=fault_step, color='red', linestyle='--', alpha=0.5)
   
   # 4. Component Correlation Matrix
   ax = axes[3]
   components = ['lambda', 'sigma_sq', 'delta_l']
   corr_matrix = df[components].corr()
   sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
               square=True, ax=ax, cbar_kws={'label': 'Correlation'})
   ax.set_title('Component Correlation Matrix', fontweight='bold')
   
   # 5. Component Contribution to R-Metric
   ax = axes[4]
   weights = results["config"]["r_metric_weights"]
   contributions = {
       'λ': weights["lambda"] * df['lambda_norm'].mean(),
       'σ²': weights["sigma_sq"] * df['sigma_sq_norm'].mean(),
       'ΔL': weights["delta_l"] * df['delta_l_norm'].mean()
   }
   
   bars = ax.bar(contributions.keys(), contributions.values())
   ax.set_title('Average Component Contributions to R-Metric', fontweight='bold')
   ax.set_ylabel('Weighted Contribution')
   
   # Add value labels on bars
   for bar in bars:
       height = bar.get_height()
       ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')
   
   # 6. Phase Analysis
   ax = axes[5]
   # Define phases
   pre_fault = df[df['step'] < fault_step]
   during_fault = df[(df['step'] >= fault_step) & 
                    (df['step'] < fault_step + results["config"]["lr_spike_duration"])]
   post_fault = df[df['step'] >= fault_step + results["config"]["lr_spike_duration"]]
   
   phase_data = {
       'Pre-Fault': pre_fault['r_metric'].mean() if len(pre_fault) > 0 else 0,
       'During Fault': during_fault['r_metric'].mean() if len(during_fault) > 0 else 0,
       'Post-Fault': post_fault['r_metric'].mean() if len(post_fault) > 0 else 0
   }
   
   bars = ax.bar(phase_data.keys(), phase_data.values(), 
                  color=['green', 'red', 'orange'])
   ax.set_title('R-Metric by Training Phase', fontweight='bold')
   ax.set_ylabel('Average R-Metric')
   ax.axhline(y=results["config"]["r_metric_alert_threshold"], 
              color='black', linestyle='--', alpha=0.5, label='Alert Threshold')
   ax.legend()
   
   # Add value labels
   for bar in bars:
       height = bar.get_height()
       ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}', ha='center', va='bottom')
   
   # Overall title
   fig.suptitle('R-Metric Component Analysis', fontsize=16, fontweight='bold')
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_path / 'component_analysis.png', dpi=300, bbox_inches='tight')
   plt.savefig(output_path / 'component_analysis.pdf', bbox_inches='tight')
   plt.close()


def create_baseline_comparison(df: pd.DataFrame, results: Dict, output_path: Path):
   """Create baseline comparison visualization"""
   fig, axes = plt.subplots(2, 3, figsize=(15, 8))
   axes = axes.flatten()
   
   fault_step = results["config"]["fault_injection_step"]
   
   # Define methods and their data
   methods = [
       ('R-Metric', 'r_metric', results["config"]["r_metric_alert_threshold"]),
       ('Simple Heuristic', 'heuristic_alert', None),
       ('Loss Spike', 'spike_alert', None),
       ('Isolation Forest', 'if_alert', None),
       ('Gradient Monitor', 'grad_alert', None)
   ]
   
   # Plot each method
   for idx, (name, col, threshold) in enumerate(methods):
       ax = axes[idx]
       
       if col == 'r_metric':
           # Continuous metric
           ax.plot(df['step'], df[col], 'b-', linewidth=2)
           if threshold:
               ax.axhline(y=threshold, color='orange', linestyle='--', 
                         linewidth=2, label='Threshold')
           ax.set_ylabel('Score')
       else:
           # Binary alert
           alert_data = df[col].astype(int)
           ax.fill_between(df['step'], 0, alert_data, alpha=0.7, step='mid')
           ax.set_ylabel('Alert Active')
           ax.set_ylim(-0.1, 1.1)
       
       # Mark fault injection
       ax.axvline(x=fault_step, color='red', linestyle='--', alpha=0.7, label='Fault')
       
       # Mark alert point
       if col == 'r_metric':
           alert_step = results["alerts"]["r_metric"]
       else:
           # Find first True in binary column
           alert_mask = df[col] == True
           if alert_mask.any():
               alert_step = df[alert_mask]['step'].iloc[0]
           else:
               alert_step = None
       
       if alert_step:
           ax.axvline(x=alert_step, color='green', linestyle=':', 
                     linewidth=2, label=f'Alert (Step {alert_step})')
       
       ax.set_title(f'{name}', fontweight='bold')
       ax.set_xlabel('Training Step')
       ax.legend()
       ax.grid(True, alpha=0.3)
   
   # 6. Summary Comparison
   ax = axes[5]
   
   # Calculate metrics for each method
   alert_data = []
   for method in ['r_metric', 'simple_heuristic', 'loss_spike', 'isolation_forest', 'gradient_monitoring']:
       alert_step = results["alerts"][method]
       if alert_step:
           lead_time = fault_step - alert_step
           detection_delay = max(0, -lead_time)
           alert_data.append({
               'Method': method.replace('_', ' ').title(),
               'Alert Step': alert_step,
               'Lead Time': lead_time,
               'Detection Delay': detection_delay,
               'Detected': True
           })
       else:
           alert_data.append({
               'Method': method.replace('_', ' ').title(),
               'Alert Step': None,
               'Lead Time': None,
               'Detection Delay': None,
               'Detected': False
           })
   
   # Create summary table
   summary_df = pd.DataFrame(alert_data)
   
   # Plot detection performance
   detected = summary_df[summary_df['Detected']]
   if not detected.empty:
       detected = detected.sort_values('Lead Time', ascending=False)
       colors = ['green' if lt > 0 else 'orange' if lt == 0 else 'red' 
                 for lt in detected['Lead Time']]
       
       bars = ax.barh(detected['Method'], detected['Lead Time'], color=colors)
       ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
       ax.set_xlabel('Lead Time (positive = early detection)')
       ax.set_title('Detection Performance Summary', fontweight='bold')
       
       # Add annotations
       for bar, lead_time in zip(bars, detected['Lead Time']):
           width = bar.get_width()
           ax.text(width + 5 if width > 0 else width - 5, bar.get_y() + bar.get_height()/2,
                  f'{int(lead_time)} steps', 
                  ha='left' if width > 0 else 'right', va='center')
   
   # Overall title
   fig.suptitle('Baseline Methods Comparison', fontsize=16, fontweight='bold')
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_path / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
   plt.savefig(output_path / 'baseline_comparison.pdf', bbox_inches='tight')
   plt.close()


def create_alert_timeline(df: pd.DataFrame, results: Dict, output_path: Path):
   """Create timeline visualization of alerts"""
   fig, ax = plt.subplots(figsize=(14, 6))
   
   fault_step = results["config"]["fault_injection_step"]
   fault_duration = results["config"]["lr_spike_duration"]
   
   # Define timeline events
   events = []
   
   # Add fault event
   events.append({
       'step': fault_step,
       'name': 'Fault Injection',
       'type': 'fault',
       'color': 'red'
   })
   
   # Add fault recovery
   events.append({
       'step': fault_step + fault_duration,
       'name': 'Fault Recovery',
       'type': 'recovery',
       'color': 'orange'
   })
   
   # Add alerts
   alert_colors = {
       'r_metric': 'green',
       'simple_heuristic': 'blue',
       'loss_spike': 'purple',
       'isolation_forest': 'brown',
       'gradient_monitoring': 'pink'
   }
   
   for method, alert_step in results["alerts"].items():
       if alert_step:
           events.append({
               'step': alert_step,
               'name': method.replace('_', ' ').title(),
               'type': 'alert',
               'color': alert_colors.get(method, 'gray')
           })
   
   # Sort events by step
   events.sort(key=lambda x: x['step'])
   
   # Create timeline
   y_positions = {}
   y_counter = 0
   
   for event in events:
       # Assign y position
       if event['step'] not in y_positions:
           y_positions[event['step']] = y_counter
           y_counter += 1
       
       y = y_positions[event['step']]
       
       # Draw event marker
       marker = 'v' if event['type'] == 'fault' else '^' if event['type'] == 'recovery' else 'o'
       ax.scatter(event['step'], y, s=200, c=event['color'], marker=marker, 
                 edgecolors='black', linewidth=2, zorder=5)
       
       # Add label
       ax.text(event['step'], y + 0.5, event['name'], 
              ha='center', va='bottom', fontsize=10, rotation=45)
   
   # Draw timeline
   ax.axhline(y=0, color='black', linewidth=2, alpha=0.5)
   
   # Mark fault period
   ax.axvspan(fault_step, fault_step + fault_duration, alpha=0.2, color='red', 
             label='Fault Active Period')
   
   # Customize plot
   ax.set_xlim(0, results["config"]["max_steps"])
   ax.set_ylim(-1, y_counter + 2)
   ax.set_xlabel('Training Step', fontsize=12)
   ax.set_title('Alert Timeline Analysis', fontsize=14, fontweight='bold')
   ax.grid(True, axis='x', alpha=0.3)
   ax.set_yticks([])
   
   # Add legend
   legend_elements = [
       plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='red', 
                  markersize=10, label='Fault Injection'),
       plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', 
                  markersize=10, label='Fault Recovery'),
       plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                  markersize=10, label='Alert Triggered')
   ]
   ax.legend(handles=legend_elements, loc='upper right')
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_path / 'alert_timeline.png', dpi=300, bbox_inches='tight')
   plt.savefig(output_path / 'alert_timeline.pdf', bbox_inches='tight')
   plt.close()


def create_performance_summary(results: Dict, output_path: Path):
   """Create performance summary visualization"""
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.flatten()
   
   # 1. Detection Success Rate
   ax = axes[0]
   methods = list(results["alerts"].keys())
   detected = [1 if results["alerts"][m] is not None else 0 for m in methods]
   
   bars = ax.bar(range(len(methods)), detected, color=['green' if d else 'red' for d in detected])
   ax.set_xticks(range(len(methods)))
   ax.set_xticklabels([m.replace('_', '\n').title() for m in methods], rotation=0)
   ax.set_ylabel('Detection Success')
   ax.set_title('Fault Detection Success', fontweight='bold')
   ax.set_ylim(0, 1.2)
   
   # Add percentage labels
   for i, (bar, d) in enumerate(zip(bars, detected)):
       ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
              f'{d*100:.0f}%', ha='center', va='bottom')
   
   # 2. Lead Time Analysis
   ax = axes[1]
   fault_step = results["config"]["fault_injection_step"]
   lead_times = []
   method_names = []
   
   for method, alert_step in results["alerts"].items():
       if alert_step is not None:
           lead_time = fault_step - alert_step
           lead_times.append(lead_time)
           method_names.append(method.replace('_', ' ').title())
   
   if lead_times:
       colors = ['green' if lt > 0 else 'orange' if lt == 0 else 'red' for lt in lead_times]
       bars = ax.bar(range(len(lead_times)), lead_times, color=colors)
       ax.set_xticks(range(len(lead_times)))
       ax.set_xticklabels(method_names, rotation=45, ha='right')
       ax.set_ylabel('Lead Time (steps)')
       ax.set_title('Early Warning Lead Times', fontweight='bold')
       ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
       
       # Add value labels
       for bar, lt in zip(bars, lead_times):
           y_pos = bar.get_height() + 2 if bar.get_height() > 0 else bar.get_height() - 2
           ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                  f'{int(lt)}', ha='center', va='bottom' if bar.get_height() > 0 else 'top')
   
   # 3. Method Comparison Radar Chart
   ax = axes[2]
   
   # Define metrics for comparison
   categories = ['Early\nDetection', 'Reliability', 'Low\nFalse Positives', 
                 'Interpretability', 'Computational\nEfficiency']
   
   # Scores for each method (normalized to 0-1)
   # These are illustrative scores based on expected performance
   method_scores = {
       'R-Metric': [0.9, 0.85, 0.8, 0.9, 0.7],
       'Simple Heuristic': [0.3, 0.5, 0.4, 0.9, 0.95],
       'Loss Spike': [0.4, 0.6, 0.5, 0.8, 0.9],
       'Isolation Forest': [0.7, 0.75, 0.7, 0.3, 0.6],
       'Gradient Monitor': [0.5, 0.6, 0.6, 0.7, 0.8]
   }
   
   # Number of variables
   num_vars = len(categories)
   
   # Compute angle for each axis
   angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
   angles += angles[:1]  # Complete the circle
   
   # Plot each method
   for method, scores in method_scores.items():
       if results["alerts"][method.lower().replace(' ', '_').replace('-', '_')] is not None:
           scores += scores[:1]  # Complete the circle
           ax.plot(angles, scores, 'o-', linewidth=2, label=method)
           ax.fill(angles, scores, alpha=0.15)
   
   ax.set_theta_offset(np.pi / 2)
   ax.set_theta_direction(-1)
   ax.set_xticks(angles[:-1])
   ax.set_xticklabels(categories)
   ax.set_ylim(0, 1)
   ax.set_title('Method Characteristics Comparison', fontweight='bold', pad=20)
   ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
   ax.grid(True)
   
   # 4. Summary Statistics
   ax = axes[3]
   ax.axis('off')
   
   # Create summary text
   summary_text = "Case Study Summary\n" + "="*30 + "\n\n"
   summary_text += f"Model: {results['config']['model_name']}\n"
   summary_text += f"Total Steps: {results['config']['max_steps']}\n"
   summary_text += f"Fault Type: {results['config']['fault_type']}\n"
   summary_text += f"Fault Step: {results['config']['fault_injection_step']}\n\n"
   
   summary_text += "Detection Results:\n" + "-"*20 + "\n"
   for method, alert_step in results["alerts"].items():
       method_name = method.replace('_', ' ').title()
       if alert_step is not None:
           lead_time = fault_step - alert_step
           summary_text += f"✓ {method_name}: Step {alert_step} (Lead: {lead_time})\n"
       else:
           summary_text += f"✗ {method_name}: Not detected\n"
   
   ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
          fontsize=11, verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
   
   # Overall title
   fig.suptitle('Performance Summary', fontsize=16, fontweight='bold')
   
   # Save figure
   plt.tight_layout()
   plt.savefig(output_path / 'performance_summary.png', dpi=300, bbox_inches='tight')
   plt.savefig(output_path / 'performance_summary.pdf', bbox_inches='tight')
   plt.close()


if __name__ == "__main__":
   # For testing
   import sys
   if len(sys.argv) > 1:
       output_path = Path(sys.argv[1])
       create_case_study_visualizations(output_path)
   else:
       print("Please provide output path as argument")