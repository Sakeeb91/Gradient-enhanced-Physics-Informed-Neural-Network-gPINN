"""
Uncertainty Quantification for Production gPINN

This module provides comprehensive uncertainty quantification methods for
real-world gPINN parameter estimation, including:

- Monte Carlo Dropout
- Deep Ensemble Methods
- Bayesian Neural Networks
- Confidence Intervals
- Sensitivity Analysis
- Parameter Correlation Analysis

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import scipy.stats as stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')

@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification"""
    ensemble_size: int = 10
    mc_samples: int = 100
    confidence_levels: List[float] = None
    bootstrap_samples: int = 1000
    sensitivity_perturbation: float = 0.1
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ

class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty quantification
    """
    
    def __init__(self, model: nn.Module, config: UncertaintyConfig):
        self.model = model
        self.config = config
    
    def predict_with_uncertainty(self, x: torch.Tensor, apply_bc: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate predictions with uncertainty estimates using MC Dropout
        """
        self.model.train()  # Enable dropout
        
        predictions = []
        parameter_samples = {'nu_e': [], 'K': []}
        
        with torch.no_grad():
            for _ in range(self.config.mc_samples):
                # Forward pass with dropout enabled
                u_pred = self.model(x, apply_bc=apply_bc)
                predictions.append(u_pred.cpu().numpy())
                
                # Sample parameters
                parameter_samples['nu_e'].append(self.model.nu_e.item())
                parameter_samples['K'].append(self.model.K.item())
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        confidence_intervals = {}
        for confidence in self.config.confidence_levels:
            alpha = 1 - confidence
            lower = np.percentile(predictions, 100 * alpha/2, axis=0)
            upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
            confidence_intervals[f'{confidence:.0%}'] = {'lower': lower, 'upper': upper}
        
        # Parameter uncertainty
        param_stats = {}
        for param_name, samples in parameter_samples.items():
            param_stats[param_name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'samples': samples
            }
        
        return {
            'mean': torch.tensor(mean_pred),
            'std': torch.tensor(std_pred),
            'confidence_intervals': confidence_intervals,
            'parameter_uncertainty': param_stats,
            'raw_predictions': predictions
        }

class EnsembleUncertainty:
    """
    Deep Ensemble method for uncertainty quantification
    """
    
    def __init__(self, model_class, config: UncertaintyConfig, model_config):
        self.model_class = model_class
        self.config = config
        self.model_config = model_config
        self.ensemble = []
        self.trained = False
    
    def create_ensemble(self, device: torch.device):
        """Create ensemble of models with different initializations"""
        print(f"🔧 Creating ensemble of {self.config.ensemble_size} models...")
        
        self.ensemble = []
        for i in range(self.config.ensemble_size):
            # Set different random seed for each model
            torch.manual_seed(42 + i)
            model = self.model_class(self.model_config).to(device)
            self.ensemble.append(model)
        
        print(f"✅ Ensemble created with {len(self.ensemble)} models")
    
    def train_ensemble(self, trainer_class, data: Dict, **training_kwargs):
        """Train all models in the ensemble"""
        print("🚀 Training ensemble models...")
        
        self.trained_models = []
        self.training_histories = []
        
        for i, model in enumerate(self.ensemble):
            print(f"Training model {i+1}/{len(self.ensemble)}...")
            
            # Create trainer for this model
            trainer = trainer_class(model, self.model_config, data, **training_kwargs)
            
            # Train model
            history = trainer.train()
            
            self.trained_models.append(model)
            self.training_histories.append(history)
        
        self.trained = True
        print("✅ Ensemble training completed!")
    
    def predict_with_uncertainty(self, x: torch.Tensor, apply_bc: bool = True) -> Dict[str, torch.Tensor]:
        """Generate ensemble predictions with uncertainty"""
        if not self.trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        parameter_samples = {'nu_e': [], 'K': []}
        
        for model in self.trained_models:
            model.eval()
            with torch.no_grad():
                u_pred = model(x, apply_bc=apply_bc)
                predictions.append(u_pred.cpu().numpy())
                
                parameter_samples['nu_e'].append(model.nu_e.item())
                parameter_samples['K'].append(model.K.item())
        
        predictions = np.array(predictions)
        
        # Compute ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        confidence_intervals = {}
        for confidence in self.config.confidence_levels:
            alpha = 1 - confidence
            lower = np.percentile(predictions, 100 * alpha/2, axis=0)
            upper = np.percentile(predictions, 100 * (1 - alpha/2), axis=0)
            confidence_intervals[f'{confidence:.0%}'] = {'lower': lower, 'upper': upper}
        
        # Parameter uncertainty
        param_stats = {}
        for param_name, samples in parameter_samples.items():
            param_stats[param_name] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'samples': samples,
                'confidence_intervals': {}
            }
            
            # Parameter confidence intervals
            for confidence in self.config.confidence_levels:
                alpha = 1 - confidence
                lower = np.percentile(samples, 100 * alpha/2)
                upper = np.percentile(samples, 100 * (1 - alpha/2))
                param_stats[param_name]['confidence_intervals'][f'{confidence:.0%}'] = {
                    'lower': lower, 'upper': upper
                }
        
        return {
            'mean': torch.tensor(mean_pred),
            'std': torch.tensor(std_pred),
            'confidence_intervals': confidence_intervals,
            'parameter_uncertainty': param_stats,
            'raw_predictions': predictions,
            'ensemble_size': len(self.trained_models)
        }

class SensitivityAnalysis:
    """
    Sensitivity analysis for parameter importance and model robustness
    """
    
    def __init__(self, model: nn.Module, config: UncertaintyConfig):
        self.model = model
        self.config = config
    
    def parameter_sensitivity(self, x: torch.Tensor, parameter_ranges: Dict) -> Dict:
        """
        Analyze sensitivity to parameter variations
        """
        print("🔍 Performing parameter sensitivity analysis...")
        
        baseline_params = {
            'nu_e': self.model.nu_e.item(),
            'K': self.model.K.item()
        }
        
        self.model.eval()
        with torch.no_grad():
            baseline_prediction = self.model(x).cpu().numpy()
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            param_values = np.linspace(min_val, max_val, 50)
            predictions = []
            
            for param_val in param_values:
                # Temporarily modify parameter
                if param_name == 'nu_e':
                    original_val = self.model.log_nu_e.data.clone()
                    self.model.log_nu_e.data = torch.tensor(np.log(param_val))
                elif param_name == 'K':
                    original_val = self.model.log_K.data.clone()
                    self.model.log_K.data = torch.tensor(np.log(param_val))
                
                # Get prediction
                with torch.no_grad():
                    pred = self.model(x).cpu().numpy()
                    predictions.append(pred)
                
                # Restore original parameter
                if param_name == 'nu_e':
                    self.model.log_nu_e.data = original_val
                elif param_name == 'K':
                    self.model.log_K.data = original_val
            
            predictions = np.array(predictions)
            
            # Compute sensitivity metrics
            sensitivity_results[param_name] = {
                'parameter_values': param_values,
                'predictions': predictions,
                'baseline_value': baseline_params[param_name],
                'sensitivity_index': np.std(predictions) / np.std(param_values),
                'max_deviation': np.max(np.abs(predictions - baseline_prediction))
            }
        
        print("✅ Sensitivity analysis completed")
        return sensitivity_results
    
    def local_sensitivity(self, x: torch.Tensor) -> Dict:
        """
        Compute local sensitivity using gradients
        """
        x = x.requires_grad_(True)
        
        # Enable gradient computation for parameters
        self.model.log_nu_e.requires_grad_(True)
        self.model.log_K.requires_grad_(True)
        
        # Forward pass
        u = self.model(x)
        
        # Compute gradients
        grad_nu_e = torch.autograd.grad(u, self.model.log_nu_e, 
                                       grad_outputs=torch.ones_like(u),
                                       create_graph=True, retain_graph=True)[0]
        
        grad_K = torch.autograd.grad(u, self.model.log_K,
                                    grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        
        # Spatial gradients
        grad_x = torch.autograd.grad(u, x,
                                    grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        
        return {
            'grad_nu_e': grad_nu_e.detach().cpu().numpy(),
            'grad_K': grad_K.detach().cpu().numpy(),
            'grad_x': grad_x.detach().cpu().numpy(),
            'sensitivity_nu_e': torch.abs(grad_nu_e).mean().item(),
            'sensitivity_K': torch.abs(grad_K).mean().item()
        }

class UncertaintyVisualizer:
    """
    Comprehensive visualization for uncertainty quantification results
    """
    
    @staticmethod
    def plot_prediction_uncertainty(x: np.ndarray, uncertainty_results: Dict, 
                                   true_solution: Optional[np.ndarray] = None,
                                   save_path: Optional[str] = None):
        """Plot predictions with uncertainty bands"""
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Main prediction plot with confidence bands
        mean_pred = uncertainty_results['mean'].numpy().flatten()
        std_pred = uncertainty_results['std'].numpy().flatten()
        x_flat = x.flatten()
        
        axs[0,0].plot(x_flat, mean_pred, 'r-', linewidth=3, label='gPINN Prediction', alpha=0.8)
        
        if true_solution is not None:
            axs[0,0].plot(x_flat, true_solution.flatten(), 'b--', linewidth=2, 
                         label='True Solution', alpha=0.8)
        
        # Confidence bands
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for i, (conf_level, conf_data) in enumerate(uncertainty_results['confidence_intervals'].items()):
            axs[0,0].fill_between(x_flat, conf_data['lower'].flatten(), conf_data['upper'].flatten(),
                                 alpha=0.3, color=colors[i % len(colors)], 
                                 label=f'{conf_level} Confidence')
        
        axs[0,0].set_title('Velocity Prediction with Uncertainty Bands\n'
                          'Quantified uncertainty in fluid flow prediction', fontweight='bold')
        axs[0,0].set_xlabel('Spatial Position')
        axs[0,0].set_ylabel('Velocity')
        axs[0,0].legend()
        axs[0,0].grid(True, alpha=0.3)
        
        # Standard deviation plot
        axs[0,1].plot(x_flat, std_pred, 'purple', linewidth=2)
        axs[0,1].fill_between(x_flat, 0, std_pred, alpha=0.3, color='purple')
        axs[0,1].set_title('Prediction Uncertainty (Standard Deviation)\n'
                          'Spatial distribution of model uncertainty', fontweight='bold')
        axs[0,1].set_xlabel('Spatial Position')
        axs[0,1].set_ylabel('Standard Deviation')
        axs[0,1].grid(True, alpha=0.3)
        
        # Parameter uncertainty
        param_names = ['nu_e', 'K']
        param_labels = ['Effective Viscosity (νₑ)', 'Permeability (K)']
        
        for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            if param_name in uncertainty_results['parameter_uncertainty']:
                param_data = uncertainty_results['parameter_uncertainty'][param_name]
                samples = param_data['samples']
                
                ax = axs[1, i]
                
                # Histogram
                ax.hist(samples, bins=20, alpha=0.7, color=f'C{i}', density=True, 
                       edgecolor='black', linewidth=1)
                
                # Statistics
                mean_val = param_data['mean']
                std_val = param_data['std']
                
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.4e}')
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=2,
                          label=f'±1σ: {std_val:.4e}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=2)
                
                ax.set_title(f'{param_label} Uncertainty Distribution\n'
                           f'Inferred parameter with confidence bounds', fontweight='bold')
                ax.set_xlabel('Parameter Value')
                ax.set_ylabel('Probability Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_sensitivity_analysis(sensitivity_results: Dict, save_path: Optional[str] = None):
        """Plot sensitivity analysis results"""
        
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        param_names = list(sensitivity_results.keys())
        
        for i, param_name in enumerate(param_names):
            if i >= 2:  # Only plot first two parameters
                break
                
            data = sensitivity_results[param_name]
            param_values = data['parameter_values']
            predictions = data['predictions']
            baseline_value = data['baseline_value']
            
            # Parameter sweep plot
            ax = axs[0, i]
            
            # Plot prediction envelope
            mean_pred = np.mean(predictions, axis=1)
            std_pred = np.std(predictions, axis=1)
            
            ax.plot(param_values, mean_pred, 'b-', linewidth=2, label='Mean Prediction')
            ax.fill_between(param_values, mean_pred - std_pred, mean_pred + std_pred,
                           alpha=0.3, color='blue', label='±1σ Envelope')
            
            ax.axvline(baseline_value, color='red', linestyle='--', linewidth=2,
                      label=f'Baseline: {baseline_value:.4e}')
            
            param_label = 'Effective Viscosity (νₑ)' if param_name == 'nu_e' else 'Permeability (K)'
            ax.set_title(f'Sensitivity to {param_label}\n'
                        f'Velocity response to parameter variations', fontweight='bold')
            ax.set_xlabel(f'{param_label} Value')
            ax.set_ylabel('Velocity')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            
            # Sensitivity index
            ax2 = axs[1, i]
            sensitivity_index = data['sensitivity_index']
            max_deviation = data['max_deviation']
            
            metrics = ['Sensitivity Index', 'Max Deviation']
            values = [sensitivity_index, max_deviation]
            colors = ['skyblue', 'lightcoral']
            
            bars = ax2.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
            ax2.set_title(f'{param_label} Sensitivity Metrics\n'
                         'Quantitative measures of parameter importance', fontweight='bold')
            ax2.set_ylabel('Sensitivity Measure')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01 * height,
                        f'{value:.3e}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.show()
    
    @staticmethod
    def plot_parameter_correlation(uncertainty_results: Dict, save_path: Optional[str] = None):
        """Plot parameter correlation analysis"""
        
        param_data = uncertainty_results['parameter_uncertainty']
        
        if len(param_data) < 2:
            print("⚠️ Need at least 2 parameters for correlation analysis")
            return
        
        # Extract parameter samples
        nu_e_samples = param_data['nu_e']['samples']
        K_samples = param_data['K']['samples']
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Scatter plot
        axs[0].scatter(nu_e_samples, K_samples, alpha=0.6, s=50, c='blue', edgecolors='black')
        axs[0].set_xlabel('Effective Viscosity (νₑ)')
        axs[0].set_ylabel('Permeability (K)')
        axs[0].set_title('Parameter Correlation\n'
                        'Joint distribution of inferred parameters', fontweight='bold')
        axs[0].grid(True, alpha=0.3)
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        
        # Correlation coefficient
        correlation = np.corrcoef(nu_e_samples, K_samples)[0, 1]
        axs[0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axs[0].transAxes, bbox=dict(boxstyle="round", facecolor='white'),
                   fontweight='bold')
        
        # Individual parameter distributions
        axs[1].hist(nu_e_samples, bins=20, alpha=0.7, color='green', density=True,
                   edgecolor='black', label='νₑ Distribution')
        axs[1].set_xlabel('Effective Viscosity (νₑ)')
        axs[1].set_ylabel('Probability Density')
        axs[1].set_title('νₑ Uncertainty Distribution\n'
                        'Marginal distribution of effective viscosity', fontweight='bold')
        axs[1].grid(True, alpha=0.3)
        axs[1].set_xscale('log')
        
        axs[2].hist(K_samples, bins=20, alpha=0.7, color='orange', density=True,
                   edgecolor='black', label='K Distribution')
        axs[2].set_xlabel('Permeability (K)')
        axs[2].set_ylabel('Probability Density')
        axs[2].set_title('K Uncertainty Distribution\n'
                        'Marginal distribution of permeability', fontweight='bold')
        axs[2].grid(True, alpha=0.3)
        axs[2].set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        
        plt.show()

class ComprehensiveUncertaintyAnalysis:
    """
    Main class for comprehensive uncertainty quantification
    """
    
    def __init__(self, model: nn.Module, config: UncertaintyConfig):
        self.model = model
        self.config = config
        self.mc_dropout = MCDropoutUncertainty(model, config)
        self.sensitivity = SensitivityAnalysis(model, config)
        self.visualizer = UncertaintyVisualizer()
    
    def full_analysis(self, x: torch.Tensor, parameter_ranges: Dict,
                     true_solution: Optional[np.ndarray] = None,
                     save_dir: Optional[str] = None) -> Dict:
        """
        Perform comprehensive uncertainty analysis
        """
        print("🔬 Starting comprehensive uncertainty analysis...")
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        # 1. MC Dropout uncertainty
        print("📊 Computing MC Dropout uncertainty...")
        mc_results = self.mc_dropout.predict_with_uncertainty(x)
        
        # 2. Sensitivity analysis
        print("🔍 Performing sensitivity analysis...")
        sensitivity_results = self.sensitivity.parameter_sensitivity(x, parameter_ranges)
        local_sensitivity = self.sensitivity.local_sensitivity(x)
        
        # 3. Generate visualizations
        print("🎨 Generating visualizations...")
        
        # Prediction uncertainty plot
        self.visualizer.plot_prediction_uncertainty(
            x.cpu().numpy(), mc_results, true_solution,
            save_path=f"{save_dir}/prediction_uncertainty.png" if save_dir else None
        )
        
        # Sensitivity analysis plot
        self.visualizer.plot_sensitivity_analysis(
            sensitivity_results,
            save_path=f"{save_dir}/sensitivity_analysis.png" if save_dir else None
        )
        
        # Parameter correlation plot
        self.visualizer.plot_parameter_correlation(
            mc_results,
            save_path=f"{save_dir}/parameter_correlation.png" if save_dir else None
        )
        
        # Compile comprehensive results
        comprehensive_results = {
            'mc_dropout': mc_results,
            'sensitivity_global': sensitivity_results,
            'sensitivity_local': local_sensitivity,
            'summary_statistics': self._compute_summary_statistics(mc_results, sensitivity_results)
        }
        
        print("✅ Comprehensive uncertainty analysis completed!")
        return comprehensive_results
    
    def _compute_summary_statistics(self, mc_results: Dict, sensitivity_results: Dict) -> Dict:
        """Compute summary statistics for uncertainty analysis"""
        
        param_uncertainty = mc_results['parameter_uncertainty']
        
        summary = {
            'parameter_estimates': {},
            'prediction_uncertainty': {
                'mean_std': float(torch.mean(mc_results['std'])),
                'max_std': float(torch.max(mc_results['std'])),
                'relative_uncertainty': float(torch.mean(mc_results['std']) / torch.mean(torch.abs(mc_results['mean'])))
            },
            'sensitivity_ranking': {}
        }
        
        # Parameter estimates with confidence intervals
        for param_name, param_data in param_uncertainty.items():
            summary['parameter_estimates'][param_name] = {
                'mean': param_data['mean'],
                'std': param_data['std'],
                'cv': param_data['std'] / param_data['mean'],  # Coefficient of variation
                'confidence_68': param_data.get('confidence_intervals', {}).get('68%', {}),
                'confidence_95': param_data.get('confidence_intervals', {}).get('95%', {})
            }
        
        # Sensitivity ranking
        for param_name, sens_data in sensitivity_results.items():
            summary['sensitivity_ranking'][param_name] = {
                'sensitivity_index': sens_data['sensitivity_index'],
                'max_deviation': sens_data['max_deviation']
            }
        
        return summary

if __name__ == "__main__":
    print("🔬 Uncertainty Quantification Module for Production gPINN")
    print("This module provides comprehensive uncertainty analysis capabilities.")
    print("Use with production_gpinn.py for complete uncertainty quantification.")