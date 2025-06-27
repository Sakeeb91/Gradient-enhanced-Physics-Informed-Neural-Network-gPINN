"""
Test Script for Production gPINN System

This script tests the complete production system with a simple synthetic dataset
to verify all components work correctly before deploying on real data.

Author: Sakeeb Rahman
Date: 2025
"""

import torch
import numpy as np
import os
from datetime import datetime

# Import our modules
from production_gpinn import ModelConfig, AdaptiveProductionPINN, ProductionTrainer, RealWorldDataProcessor

def create_simple_test_data():
    """Create simple synthetic test data for system validation"""
    print("🔧 Creating simple test data...")
    
    # Simple parameters
    nu_e_true = 1e-3
    K_true = 1e-3
    nu = 1e-3
    g = 1.0
    H = 1.0
    
    # Simple analytical solution (stable version)
    def analytical_solution(x):
        return g * K_true / nu * x * (H - x)  # Simple parabolic profile
    
    # Generate test points
    n_points = 20
    x_data = np.linspace(0.05, 0.95, n_points)
    u_data = analytical_solution(x_data)
    
    # Add small amount of noise
    noise = 0.01 * np.std(u_data) * np.random.randn(n_points)
    u_data += noise
    
    # Measurement errors
    errors = np.full(n_points, 0.02)
    
    return {
        'x_data': x_data,
        'u_data': u_data,
        'errors': errors,
        'metadata': {
            'true_nu_e': nu_e_true,
            'true_K': K_true,
            'nu': nu,
            'g': g,
            'H': H
        }
    }

def test_production_system():
    """Test the complete production system"""
    print("🧪 Testing Production gPINN System")
    print("=" * 50)
    
    # Create test data
    test_data = create_simple_test_data()
    
    # Configuration for fast testing
    config = ModelConfig(
        hidden_layers=3,
        hidden_size=32,
        activation='tanh',
        lambda_physics=1.0,
        lambda_gradient=0.1,
        learning_rate=1e-3,
        epochs=5000,  # Reduced for testing
        patience=1000,
        augmentation_factor=2,
        noise_level=0.02
    )
    
    print(f"📊 Test data: {len(test_data['x_data'])} points")
    print(f"🔧 Model config: {config.hidden_layers} layers, {config.hidden_size} neurons")
    
    # Data preprocessing
    processor = RealWorldDataProcessor(config)
    processed_data = processor.preprocess(test_data)
    
    print(f"✅ Preprocessed: {len(processed_data['x_train'])} train, {len(processed_data['x_val'])} val")
    
    # Model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AdaptiveProductionPINN(config).to(device)
    
    print(f"🧠 Model created: {sum(p.numel() for p in model.parameters())} parameters on {device}")
    
    # Training
    trainer = ProductionTrainer(
        model, config, processed_data,
        nu=test_data['metadata']['nu'],
        g=test_data['metadata']['g']
    )
    
    print("🚀 Starting training...")
    history = trainer.train()
    
    # Results
    final_nu_e = model.nu_e.item()
    final_K = model.K.item()
    true_nu_e = test_data['metadata']['true_nu_e']
    true_K = test_data['metadata']['true_K']
    
    nu_e_error = abs(final_nu_e - true_nu_e) / true_nu_e * 100
    K_error = abs(final_K - true_K) / true_K * 100
    
    print("\n✅ Training completed!")
    print(f"🎯 Results:")
    print(f"   • νₑ: {final_nu_e:.4e} (true: {true_nu_e:.4e}, error: {nu_e_error:.2f}%)")
    print(f"   • K: {final_K:.4e} (true: {true_K:.4e}, error: {K_error:.2f}%)")
    print(f"   • Training epochs: {len(history['train_loss'])}")
    print(f"   • Final loss: {history['train_loss'][-1]:.4e}")
    
    # Basic validation
    success = True
    if nu_e_error > 50:  # Allow 50% error for simple test
        print("❌ νₑ estimation error too high")
        success = False
    if K_error > 50:
        print("❌ K estimation error too high") 
        success = False
    if len(history['train_loss']) >= config.epochs:
        print("⚠️ Training did not converge within epoch limit")
    
    if success:
        print("🌟 Production system test PASSED!")
    else:
        print("❌ Production system test FAILED!")
    
    return success, {
        'model': model,
        'trainer': trainer,
        'history': history,
        'errors': {'nu_e_error': nu_e_error, 'K_error': K_error}
    }

def test_data_loading():
    """Test data loading utilities"""
    print("\n🧪 Testing Data Loading Utilities")
    print("-" * 40)
    
    try:
        # Create simple CSV data
        import pandas as pd
        os.makedirs('test_data', exist_ok=True)
        
        test_df = pd.DataFrame({
            'well_id': ['W1', 'W2', 'W3', 'W4', 'W5'],
            'x_position': [0.1, 0.3, 0.5, 0.7, 0.9],
            'velocity': [0.1, 0.3, 0.4, 0.3, 0.1],
            'measurement_error': [0.02, 0.02, 0.02, 0.02, 0.02],
            'timestamp': ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01', '2024-01-01']
        })
        
        test_df.to_csv('test_data/simple_test.csv', index=False)
        
        # Test loading
        from data_utils import RealWorldDataLoader
        loader = RealWorldDataLoader(data_validation=False)  # Disable validation for simple test
        
        data = loader.load_csv_data('test_data/simple_test.csv')
        print(f"✅ CSV loading: {data['n_measurements']} measurements")
        
        # Clean up
        import shutil
        shutil.rmtree('test_data')
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Production gPINN System Test Suite")
    print("=" * 60)
    
    # Test 1: Data loading
    data_test_passed = test_data_loading()
    
    # Test 2: Production system
    system_test_passed, results = test_production_system()
    
    # Overall results
    print("\n" + "=" * 60)
    print("📊 TEST SUITE RESULTS:")
    print(f"   • Data Loading: {'✅ PASSED' if data_test_passed else '❌ FAILED'}")
    print(f"   • Production System: {'✅ PASSED' if system_test_passed else '❌ FAILED'}")
    
    if data_test_passed and system_test_passed:
        print("\n🌟 ALL TESTS PASSED! Production system is ready for real-world data!")
        print("🚀 You can now run: python run_real_world_experiment.py")
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
    
    print("\n📋 Next Steps:")
    print("   1. Prepare your real-world CSV data with columns: x_position, velocity, measurement_error")
    print("   2. Use RealWorldDataLoader to load your data")
    print("   3. Run production_gpinn.py with your data")
    print("   4. Apply uncertainty_quantification.py for comprehensive analysis")