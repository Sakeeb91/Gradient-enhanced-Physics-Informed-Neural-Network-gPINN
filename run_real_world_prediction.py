"""
Real-World gPINN Prediction System
Analyze real-world flow data using multiple gPINN implementations

This script runs the complete analysis pipeline:
1. Load real-world datasets
2. Run parameter estimation with different gPINN implementations  
3. Uncertainty quantification
4. Performance comparison and visualization

Author: Sakeeb Rahman
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import time

# Import our gPINN implementations
try:
    from numpy_gpinn import NumpyPhysicsInformedNN, NumpyGPINNConfig
    NUMPY_AVAILABLE = True
except ImportError:
    print("⚠️ NumPy gPINN not available")
    NUMPY_AVAILABLE = False

try:
    from sklearn_gpinn import SklearnPhysicsInformedNN, SklearnGPINNConfig, SklearnGPINNTrainer
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Scikit-learn gPINN not available")
    SKLEARN_AVAILABLE = False

try:
    import torch
    from production_gpinn import AdaptiveProductionPINN, ModelConfig
    PYTORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch gPINN not available")
    PYTORCH_AVAILABLE = False

class RealWorldGPINNAnalyzer:
    """Complete real-world analysis system"""
    
    def __init__(self):
        self.results = {}
        self.datasets = {}
        
    def load_datasets(self):
        """Load all available real-world datasets"""
        print("📂 Loading real-world datasets...")
        
        # Kansas aquifer data
        try:
            self.datasets['kansas'] = {
                'data': pd.read_csv('real_world_data/kansas_aquifer_flow_data.csv'),
                'metadata': json.load(open('real_world_data/kansas_aquifer_metadata.json'))
            }
            print(f"   ✅ Kansas aquifer: {len(self.datasets['kansas']['data'])} measurements")
        except Exception as e:
            print(f"   ❌ Failed to load Kansas data: {e}")
            
        # Nevada geothermal data
        try:
            self.datasets['nevada'] = {
                'data': pd.read_csv('real_world_data/nevada_geothermal_flow_data.csv'),
                'metadata': json.load(open('real_world_data/nevada_geothermal_metadata.json'))
            }
            print(f"   ✅ Nevada geothermal: {len(self.datasets['nevada']['data'])} measurements")
        except Exception as e:
            print(f"   ❌ Failed to load Nevada data: {e}")
            
        return len(self.datasets) > 0
        
    def prepare_data_for_gpinn(self, dataset_name):
        """Convert dataset to gPINN format"""
        data = self.datasets[dataset_name]['data']
        
        # Extract coordinates and measurements
        x = data['x_position_norm'].values.reshape(-1, 1)
        u_measured = data['velocity_ms'].values.reshape(-1, 1)
        measurement_errors = data['measurement_error'].values.reshape(-1, 1)
        
        # Create domain points for physics constraints
        x_physics = np.linspace(0, 1, 50).reshape(-1, 1)
        
        return {
            'x_data': x,
            'u_data': u_measured,
            'errors': measurement_errors,
            'x_physics': x_physics,
            'domain': [0, 1]
        }
        
    def run_numpy_gpinn(self, data_dict, dataset_name):
        """Run NumPy implementation"""
        if not NUMPY_AVAILABLE:
            return None
            
        print(f"   🔢 Running NumPy gPINN on {dataset_name}...")
        
        config = NumpyGPINNConfig(
            hidden_layers=[50, 50, 50],
            learning_rate=0.001,
            n_epochs=2000,
            physics_weight=1.0,
            gradient_weight=0.5
        )
        
        start_time = time.time()
        
        # Initialize and train
        model = NumpyPhysicsInformedNN(config)
        loss_history = model.fit(
            data_dict['x_data'], 
            data_dict['u_data'],
            data_dict['x_physics']
        )
        
        training_time = time.time() - start_time
        
        # Predictions
        x_pred = np.linspace(0, 1, 100).reshape(-1, 1)
        u_pred = model.predict(x_pred)
        
        # Extract estimated parameters
        K_est = model.K_estimated
        nu_e_est = model.nu_e_estimated
        
        return {
            'implementation': 'NumPy',
            'K_estimated': float(K_est),
            'nu_e_estimated': float(nu_e_est),
            'x_pred': x_pred.flatten(),
            'u_pred': u_pred.flatten(),
            'loss_history': loss_history,
            'training_time': training_time,
            'final_loss': loss_history[-1] if loss_history else None
        }
        
    def run_sklearn_gpinn(self, data_dict, dataset_name):
        """Run Scikit-learn implementation"""
        if not SKLEARN_AVAILABLE:
            return None
            
        print(f"   🔬 Running Scikit-learn gPINN on {dataset_name}...")
        
        config = SklearnGPINNConfig(
            hidden_layer_sizes=(100, 100),
            max_iter=1000,
            learning_rate_init=0.001,
            lambda_physics=1.0
        )
        
        # Prepare data in sklearn format
        from sklearn.model_selection import train_test_split
        
        x_data = data_dict['x_data'].flatten()
        u_data = data_dict['u_data'].flatten()
        
        # Split data
        x_train, x_val, u_train, u_val = train_test_split(
            x_data, u_data, test_size=0.2, random_state=42
        )
        
        sklearn_data = {
            'x_train': x_train,
            'u_train': u_train,
            'x_val': x_val,
            'u_val': u_val
        }
        
        start_time = time.time()
        
        # Initialize and train
        trainer = SklearnGPINNTrainer(config, sklearn_data, nu=1e-3, g=1.0)
        trainer.train()
        
        training_time = time.time() - start_time
        
        # Predictions
        x_pred = np.linspace(0, 1, 100).reshape(-1, 1)
        u_pred = trainer.model.forward(x_pred)
        
        # Extract estimated parameters
        K_est = trainer.model.K
        nu_e_est = trainer.model.nu_e
        
        return {
            'implementation': 'Scikit-learn',
            'K_estimated': float(K_est),
            'nu_e_estimated': float(nu_e_est),
            'x_pred': x_pred.flatten(),
            'u_pred': u_pred.flatten(),
            'training_time': training_time,
            'convergence': trainer.model.nn.n_iter_ if hasattr(trainer.model.nn, 'n_iter_') else None
        }
        
    def run_pytorch_gpinn(self, data_dict, dataset_name):
        """Run PyTorch implementation if available"""
        if not PYTORCH_AVAILABLE:
            return None
            
        print(f"   🔥 Running PyTorch gPINN on {dataset_name}...")
        
        config = ModelConfig(
            hidden_layers=[64, 64, 64],
            learning_rate=0.001,
            n_epochs=2000,
            physics_weight=1.0,
            batch_size=32
        )
        
        start_time = time.time()
        
        # Convert to PyTorch tensors
        x_data = torch.FloatTensor(data_dict['x_data'])
        u_data = torch.FloatTensor(data_dict['u_data'])
        x_physics = torch.FloatTensor(data_dict['x_physics'])
        
        # Initialize and train
        model = AdaptiveProductionPINN(config)
        loss_history = model.train_model(x_data, u_data, x_physics)
        
        training_time = time.time() - start_time
        
        # Predictions
        x_pred = torch.linspace(0, 1, 100).reshape(-1, 1)
        with torch.no_grad():
            u_pred = model(x_pred).numpy()
            K_est = model.K.item()
            nu_e_est = model.nu_e.item()
        
        return {
            'implementation': 'PyTorch',
            'K_estimated': float(K_est),
            'nu_e_estimated': float(nu_e_est),
            'x_pred': x_pred.numpy().flatten(),
            'u_pred': u_pred.flatten(),
            'loss_history': loss_history,
            'training_time': training_time,
            'final_loss': loss_history[-1] if loss_history else None
        }
        
    def calculate_uncertainty(self, data_dict, best_model_result):
        """Calculate prediction uncertainty using bootstrap sampling"""
        print("   📊 Calculating uncertainty estimates...")
        
        n_bootstrap = 50
        x_pred = best_model_result['x_pred']
        predictions = []
        
        # Bootstrap sampling
        n_data = len(data_dict['x_data'])
        
        for i in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_data, n_data, replace=True)
            x_boot = data_dict['x_data'][indices]
            u_boot = data_dict['u_data'][indices]
            
            # Add noise based on measurement errors
            noise = np.random.normal(0, data_dict['errors'][indices])
            u_boot_noisy = u_boot + noise
            
            # Quick prediction (simplified for demonstration)
            # In practice, would retrain model on bootstrap sample
            u_pred_boot = np.interp(x_pred, x_boot.flatten(), u_boot_noisy.flatten())
            predictions.append(u_pred_boot)
        
        predictions = np.array(predictions)
        
        # Calculate confidence intervals
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
        
    def analyze_dataset(self, dataset_name):
        """Complete analysis of a single dataset"""
        print(f"\n🔬 Analyzing {dataset_name} dataset...")
        print("=" * 50)
        
        if dataset_name not in self.datasets:
            print(f"   ❌ Dataset {dataset_name} not available")
            return None
            
        # Prepare data
        data_dict = self.prepare_data_for_gpinn(dataset_name)
        
        # Get true parameters if available
        true_params = self.datasets[dataset_name]['metadata'].get('true_parameters', {})
        K_true = true_params.get('K_true', None)
        nu_e_true = true_params.get('nu_e_true', None)
        
        print(f"   📋 True parameters (if known):")
        if K_true is not None:
            print(f"      • K_true = {K_true:.3e} m/s")
        if nu_e_true is not None:
            print(f"      • νₑ_true = {nu_e_true:.3e} Pa·s")
        
        # Run all available implementations
        results = {}
        
        if NUMPY_AVAILABLE:
            results['numpy'] = self.run_numpy_gpinn(data_dict, dataset_name)
            
        if SKLEARN_AVAILABLE:
            results['sklearn'] = self.run_sklearn_gpinn(data_dict, dataset_name)
            
        if PYTORCH_AVAILABLE:
            results['pytorch'] = self.run_pytorch_gpinn(data_dict, dataset_name)
        
        # Find best result for uncertainty analysis
        best_result = None
        best_score = float('inf')
        
        for impl_name, result in results.items():
            if result and result.get('final_loss', float('inf')) < best_score:
                best_result = result
                best_score = result.get('final_loss', float('inf'))
        
        # Calculate uncertainty
        uncertainty = None
        if best_result:
            uncertainty = self.calculate_uncertainty(data_dict, best_result)
        
        # Store results
        analysis_result = {
            'dataset_name': dataset_name,
            'data_dict': data_dict,
            'true_parameters': true_params,
            'implementations': results,
            'best_result': best_result,
            'uncertainty': uncertainty,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.results[dataset_name] = analysis_result
        
        # Print summary
        print(f"\n   📊 Analysis Summary for {dataset_name}:")
        for impl_name, result in results.items():
            if result:
                print(f"      {result['implementation']}:")
                print(f"         • K_est = {result['K_estimated']:.3e} m/s")
                print(f"         • νₑ_est = {result['nu_e_estimated']:.3e} Pa·s")
                print(f"         • Training time: {result['training_time']:.1f}s")
                
                # Calculate errors if true values known
                if K_true is not None:
                    K_error = abs(result['K_estimated'] - K_true) / K_true * 100
                    print(f"         • K error: {K_error:.1f}%")
                if nu_e_true is not None:
                    nu_e_error = abs(result['nu_e_estimated'] - nu_e_true) / nu_e_true * 100
                    print(f"         • νₑ error: {nu_e_error:.1f}%")
        
        return analysis_result
        
    def create_analysis_visualization(self, dataset_name):
        """Create comprehensive visualization of analysis results"""
        if dataset_name not in self.results:
            return
            
        result = self.results[dataset_name]
        data_dict = result['data_dict']
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Data and predictions
        ax = axs[0, 0]
        
        # Original data points
        ax.errorbar(data_dict['x_data'].flatten(), data_dict['u_data'].flatten(),
                   yerr=data_dict['errors'].flatten(), fmt='o', color='black',
                   label='Measurements', capsize=3, markersize=8)
        
        # Predictions from different implementations
        colors = {'numpy': 'blue', 'sklearn': 'green', 'pytorch': 'red'}
        
        for impl_name, impl_result in result['implementations'].items():
            if impl_result:
                color = colors.get(impl_name, 'gray')
                ax.plot(impl_result['x_pred'], impl_result['u_pred'], 
                       color=color, linewidth=2, 
                       label=f"{impl_result['implementation']} gPINN")
        
        # Uncertainty bands
        if result['uncertainty']:
            unc = result['uncertainty']
            best = result['best_result']
            ax.fill_between(best['x_pred'], unc['ci_lower'], unc['ci_upper'],
                           alpha=0.2, color='gray', label='95% CI')
        
        ax.set_xlabel('Normalized Position')
        ax.set_ylabel('Flow Velocity [m/s]')
        ax.set_title(f'{dataset_name.title()} Dataset: gPINN Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Parameter estimation comparison
        ax = axs[0, 1]
        
        implementations = []
        K_estimates = []
        nu_e_estimates = []
        
        for impl_name, impl_result in result['implementations'].items():
            if impl_result:
                implementations.append(impl_result['implementation'])
                K_estimates.append(impl_result['K_estimated'])
                nu_e_estimates.append(impl_result['nu_e_estimated'])
        
        x_pos = np.arange(len(implementations))
        
        # Plot K estimates
        ax.bar(x_pos - 0.2, K_estimates, 0.4, label='K estimates', alpha=0.7)
        
        # Add true values if available
        true_params = result['true_parameters']
        if 'K_true' in true_params:
            ax.axhline(y=true_params['K_true'], color='red', linestyle='--', 
                      linewidth=2, label='K true')
        
        ax.set_xlabel('Implementation')
        ax.set_ylabel('Permeability K [m/s]')
        ax.set_title('Parameter Estimation: Permeability')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(implementations)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Effective viscosity estimates  
        ax = axs[1, 0]
        
        ax.bar(x_pos, nu_e_estimates, 0.4, label='νₑ estimates', alpha=0.7, color='orange')
        
        if 'nu_e_true' in true_params:
            ax.axhline(y=true_params['nu_e_true'], color='red', linestyle='--',
                      linewidth=2, label='νₑ true')
        
        ax.set_xlabel('Implementation')
        ax.set_ylabel('Effective Viscosity νₑ [Pa·s]')
        ax.set_title('Parameter Estimation: Effective Viscosity')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(implementations)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Training convergence
        ax = axs[1, 1]
        
        for impl_name, impl_result in result['implementations'].items():
            if impl_result and 'loss_history' in impl_result and impl_result['loss_history']:
                loss_history = impl_result['loss_history']
                color = colors.get(impl_name, 'gray')
                ax.semilogy(loss_history, color=color, linewidth=2,
                           label=f"{impl_result['implementation']}")
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        vis_filename = f'real_world_data/{dataset_name}_gpinn_analysis.png'
        plt.savefig(vis_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"   📊 Analysis visualization saved: {vis_filename}")
        
    def create_summary_report(self):
        """Create comprehensive summary report"""
        print("\n📋 Creating comprehensive analysis report...")
        
        report = {
            'analysis_date': datetime.now().isoformat(),
            'datasets_analyzed': list(self.results.keys()),
            'implementations_tested': [],
            'summary_statistics': {},
            'recommendations': []
        }
        
        # Collect implementation info
        for dataset_name, result in self.results.items():
            for impl_name in result['implementations'].keys():
                if impl_name not in report['implementations_tested']:
                    report['implementations_tested'].append(impl_name)
        
        # Performance summary
        for dataset_name, result in self.results.items():
            dataset_summary = {
                'n_measurements': len(result['data_dict']['x_data']),
                'domain_size': result['data_dict']['domain'],
                'implementations': {}
            }
            
            # True parameters
            true_params = result['true_parameters']
            if true_params:
                dataset_summary['true_parameters'] = true_params
            
            # Implementation results
            for impl_name, impl_result in result['implementations'].items():
                if impl_result:
                    impl_summary = {
                        'K_estimated': impl_result['K_estimated'],
                        'nu_e_estimated': impl_result['nu_e_estimated'],
                        'training_time': impl_result['training_time']
                    }
                    
                    # Calculate errors if true values available
                    if 'K_true' in true_params:
                        K_error = abs(impl_result['K_estimated'] - true_params['K_true']) / true_params['K_true']
                        impl_summary['K_relative_error'] = float(K_error)
                    
                    if 'nu_e_true' in true_params:
                        nu_e_error = abs(impl_result['nu_e_estimated'] - true_params['nu_e_true']) / true_params['nu_e_true']
                        impl_summary['nu_e_relative_error'] = float(nu_e_error)
                    
                    dataset_summary['implementations'][impl_name] = impl_summary
            
            report['summary_statistics'][dataset_name] = dataset_summary
        
        # Generate recommendations
        if len(report['implementations_tested']) > 1:
            report['recommendations'].append("Multiple implementations tested successfully")
        
        if NUMPY_AVAILABLE:
            report['recommendations'].append("NumPy implementation provides good portability")
        
        if PYTORCH_AVAILABLE:
            report['recommendations'].append("PyTorch implementation offers advanced features")
        
        # Save report
        report_filename = 'real_world_data/gpinn_analysis_report.json'
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"   📄 Analysis report saved: {report_filename}")
        
        return report
        
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("🚀 Starting Complete Real-World gPINN Analysis")
        print("=" * 60)
        
        # Load datasets
        if not self.load_datasets():
            print("❌ No datasets available for analysis")
            return
        
        # Analyze each dataset
        for dataset_name in self.datasets.keys():
            self.analyze_dataset(dataset_name)
            self.create_analysis_visualization(dataset_name)
        
        # Create summary report
        report = self.create_summary_report()
        
        print("\n🎉 Complete Real-World Analysis Finished!")
        print("=" * 60)
        print(f"📊 Analyzed {len(self.results)} datasets")
        print(f"🔧 Tested {len(report['implementations_tested'])} implementations")
        print("📁 Results saved in real_world_data/ directory")
        
        return self.results, report

def main():
    """Main analysis execution"""
    # Create results directory
    os.makedirs('real_world_data', exist_ok=True)
    
    # Initialize analyzer
    analyzer = RealWorldGPINNAnalyzer()
    
    # Run complete analysis
    results, report = analyzer.run_complete_analysis()
    
    return results, report

if __name__ == "__main__":
    results, report = main()