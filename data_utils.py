"""
Data Utilities for Real-World gPINN Applications

This module provides utilities for loading, preprocessing, and validating
real-world geothermal/hydrogeological sensor data for gPINN training.

Supports:
- CSV data from well monitoring systems
- JSON data from IoT sensor networks
- Excel files from field measurement campaigns
- Real-time data streams
- Multi-well spatial data

Author: Sakeeb Rahman
Date: 2025
"""

import pandas as pd
import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class WellData:
    """Structure for individual well data"""
    well_id: str
    x_position: float
    y_position: Optional[float] = None
    z_position: Optional[float] = None
    measurements: List[Dict] = None
    metadata: Dict = None

@dataclass
class FieldCampaign:
    """Structure for complete field measurement campaign"""
    campaign_id: str
    site_name: str
    wells: List[WellData]
    environmental_conditions: Dict
    measurement_protocol: Dict
    data_quality_flags: Dict

class RealWorldDataLoader:
    """
    Comprehensive data loader for real-world geothermal/hydrogeological data
    """
    
    def __init__(self, data_validation: bool = True):
        self.data_validation = data_validation
        self.supported_formats = ['.csv', '.json', '.xlsx', '.pkl']
        
    def load_csv_data(self, filepath: str, config: Dict = None) -> Dict:
        """
        Load data from CSV files (most common format for well data)
        
        Expected CSV format:
        well_id, x_position, velocity, measurement_error, timestamp, temperature, pressure
        """
        print(f"📊 Loading CSV data from: {filepath}")
        
        try:
            df = pd.read_csv(filepath)
            
            # Default column mapping
            default_config = {
                'well_id_col': 'well_id',
                'x_position_col': 'x_position', 
                'velocity_col': 'velocity',
                'error_col': 'measurement_error',
                'timestamp_col': 'timestamp',
                'temperature_col': 'temperature',
                'pressure_col': 'pressure'
            }
            
            if config:
                default_config.update(config)
            
            # Validate required columns
            required_cols = [default_config['x_position_col'], default_config['velocity_col']]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract and process data
            data = {
                'x_data': df[default_config['x_position_col']].values,
                'u_data': df[default_config['velocity_col']].values,
                'source': 'csv',
                'filepath': filepath,
                'n_measurements': len(df)
            }
            
            # Add optional columns if present
            if default_config['error_col'] in df.columns:
                data['errors'] = df[default_config['error_col']].values
            else:
                # Estimate errors as 5% of velocity magnitude
                data['errors'] = 0.05 * np.abs(data['u_data'])
            
            if default_config['well_id_col'] in df.columns:
                data['well_ids'] = df[default_config['well_id_col']].values
            
            if default_config['timestamp_col'] in df.columns:
                data['timestamps'] = pd.to_datetime(df[default_config['timestamp_col']]).values
            
            if default_config['temperature_col'] in df.columns:
                data['temperatures'] = df[default_config['temperature_col']].values
                
            if default_config['pressure_col'] in df.columns:
                data['pressures'] = df[default_config['pressure_col']].values
            
            print(f"✅ Loaded {len(data['x_data'])} measurements from {len(df[default_config['well_id_col']].unique()) if default_config['well_id_col'] in df.columns else 'unknown'} wells")
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            raise
        
        return self._validate_and_clean_data(data) if self.data_validation else data
    
    def load_json_data(self, filepath: str) -> Dict:
        """
        Load data from JSON files (common for IoT sensor networks)
        
        Expected JSON format:
        {
            "campaign_info": {...},
            "wells": [
                {
                    "well_id": "W001",
                    "position": {"x": 100.0, "y": 200.0},
                    "measurements": [
                        {"timestamp": "2024-01-01T10:00:00", "velocity": 0.5, "error": 0.02}
                    ]
                }
            ]
        }
        """
        print(f"📊 Loading JSON data from: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            
            # Extract measurements from all wells
            x_data = []
            u_data = []
            errors = []
            well_ids = []
            timestamps = []
            
            for well in json_data.get('wells', []):
                well_id = well.get('well_id', 'unknown')
                x_pos = well.get('position', {}).get('x', 0.0)
                
                for measurement in well.get('measurements', []):
                    x_data.append(x_pos)
                    u_data.append(measurement.get('velocity', 0.0))
                    errors.append(measurement.get('error', 0.05 * abs(measurement.get('velocity', 0.0))))
                    well_ids.append(well_id)
                    
                    if 'timestamp' in measurement:
                        timestamps.append(pd.to_datetime(measurement['timestamp']))
            
            data = {
                'x_data': np.array(x_data),
                'u_data': np.array(u_data),
                'errors': np.array(errors),
                'well_ids': np.array(well_ids),
                'source': 'json',
                'filepath': filepath,
                'n_measurements': len(x_data),
                'campaign_info': json_data.get('campaign_info', {})
            }
            
            if timestamps:
                data['timestamps'] = np.array(timestamps)
            
            print(f"✅ Loaded {len(x_data)} measurements from {len(set(well_ids))} wells")
            
        except Exception as e:
            print(f"❌ Error loading JSON data: {e}")
            raise
        
        return self._validate_and_clean_data(data) if self.data_validation else data
    
    def load_excel_data(self, filepath: str, sheet_name: str = 'Data') -> Dict:
        """
        Load data from Excel files (common for field campaign reports)
        """
        print(f"📊 Loading Excel data from: {filepath}")
        
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Convert to CSV-like format and use existing CSV loader logic
            data = {
                'x_data': df.iloc[:, 0].values,  # Assume first column is position
                'u_data': df.iloc[:, 1].values,  # Assume second column is velocity
                'source': 'excel',
                'filepath': filepath,
                'n_measurements': len(df)
            }
            
            # Add error column if present
            if df.shape[1] > 2:
                data['errors'] = df.iloc[:, 2].values
            else:
                data['errors'] = 0.05 * np.abs(data['u_data'])
            
            print(f"✅ Loaded {len(data['x_data'])} measurements from Excel file")
            
        except Exception as e:
            print(f"❌ Error loading Excel data: {e}")
            raise
        
        return self._validate_and_clean_data(data) if self.data_validation else data
    
    def load_data(self, filepath: str, **kwargs) -> Dict:
        """
        Universal data loader that detects format and loads appropriately
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            return self.load_csv_data(filepath, kwargs.get('config'))
        elif file_ext == '.json':
            return self.load_json_data(filepath)
        elif file_ext in ['.xlsx', '.xls']:
            return self.load_excel_data(filepath, kwargs.get('sheet_name', 'Data'))
        elif file_ext == '.pkl':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}")
    
    def _validate_and_clean_data(self, data: Dict) -> Dict:
        """
        Validate and clean real-world data
        """
        print("🔍 Validating and cleaning data...")
        
        x_data = np.array(data['x_data'])
        u_data = np.array(data['u_data'])
        errors = np.array(data.get('errors', np.ones_like(u_data) * 0.05))
        
        original_size = len(x_data)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x_data) | np.isnan(u_data) | np.isnan(errors))
        x_data = x_data[valid_mask]
        u_data = u_data[valid_mask]
        errors = errors[valid_mask]
        
        # Remove obvious outliers (beyond 3 sigma)
        u_mean = np.mean(u_data)
        u_std = np.std(u_data)
        outlier_mask = np.abs(u_data - u_mean) < 3 * u_std
        x_data = x_data[outlier_mask]
        u_data = u_data[outlier_mask]
        errors = errors[outlier_mask]
        
        # Check for minimum data requirements
        if len(x_data) < 5:
            raise ValueError(f"Insufficient data points: {len(x_data)} (minimum 5 required)")
        
        # Check spatial coverage
        x_range = np.max(x_data) - np.min(x_data)
        if x_range < 0.1:
            print("⚠️  Warning: Limited spatial coverage detected")
        
        # Update other arrays if present
        for key in ['well_ids', 'timestamps', 'temperatures', 'pressures']:
            if key in data:
                data[key] = np.array(data[key])[valid_mask][outlier_mask]
        
        # Update data
        data.update({
            'x_data': x_data,
            'u_data': u_data,
            'errors': errors,
            'n_measurements': len(x_data),
            'data_quality': {
                'original_points': original_size,
                'cleaned_points': len(x_data),
                'removal_rate': (original_size - len(x_data)) / original_size * 100,
                'spatial_range': x_range,
                'velocity_range': (np.min(u_data), np.max(u_data)),
                'mean_error': np.mean(errors)
            }
        })
        
        print(f"✅ Data validation complete: {len(x_data)}/{original_size} points retained")
        return data

class RealWorldDataGenerator:
    """
    Generate realistic synthetic datasets that mimic real-world conditions
    """
    
    @staticmethod
    def generate_geothermal_campaign(n_wells: int = 12, campaign_duration_days: int = 30) -> Dict:
        """
        Generate realistic geothermal field campaign data
        """
        print(f"🌋 Generating realistic geothermal campaign data ({n_wells} wells, {campaign_duration_days} days)")
        
        # Realistic geothermal reservoir parameters
        reservoir_depth = np.random.uniform(1500, 3000)  # meters
        reservoir_temp = np.random.uniform(150, 250)     # Celsius
        nu_e_true = np.random.uniform(5e-4, 2e-3)       # Effective viscosity
        K_true = np.random.uniform(5e-4, 2e-3)          # Permeability
        
        # Well positioning (realistic spatial distribution)
        domain_length = 2000  # meters
        well_positions = np.sort(np.random.uniform(0.1, 0.9, n_wells)) * domain_length
        
        # Generate measurements over time
        timestamps = []
        x_data = []
        u_data = []
        errors = []
        well_ids = []
        temperatures = []
        pressures = []
        
        # Base analytical solution
        nu = 1e-3  # Water viscosity
        g = np.random.uniform(0.8, 1.5)  # Pressure gradient
        
        def analytical_solution(x, nu_e, K, nu, g, H):
            r = np.sqrt(nu / (nu_e * K))
            # Prevent numerical overflow by clipping large values
            r_clipped = np.clip(r, 0, 10)  # Limit r to prevent overflow
            arg1 = r_clipped * (x - H/2)
            arg2 = r_clipped * H/2
            
            # Use stable computation for large arguments
            cosh_arg1 = np.cosh(np.clip(arg1, -50, 50))
            cosh_arg2 = np.cosh(np.clip(arg2, -50, 50))
            
            # Avoid division by zero
            ratio = np.where(cosh_arg2 > 1e-10, cosh_arg1 / cosh_arg2, 1.0)
            
            return g * K / nu * (1 - ratio)
        
        # Generate time series for each well
        start_date = datetime.now() - timedelta(days=campaign_duration_days)
        
        for day in range(campaign_duration_days):
            current_date = start_date + timedelta(days=day)
            
            # Daily variations (temperature, pressure, etc.)
            daily_temp_factor = 1 + 0.05 * np.sin(2 * np.pi * day / 365)  # Seasonal
            daily_pressure_factor = 1 + 0.03 * np.sin(2 * np.pi * day / 30)  # Monthly
            
            for i, x_pos in enumerate(well_positions):
                well_id = f"GW-{i+1:03d}"
                
                # Base velocity from physics
                u_base = analytical_solution(x_pos, nu_e_true, K_true, nu, g, domain_length)
                
                # Add realistic effects
                # 1. Temperature dependence
                temp_effect = daily_temp_factor * (1 - 0.1 * abs(x_pos - domain_length/2) / (domain_length/2))
                
                # 2. Pressure variations
                pressure_effect = daily_pressure_factor
                
                # 3. Well-specific characteristics
                well_efficiency = np.random.uniform(0.85, 1.15)  # Well condition factor
                
                # 4. Measurement noise
                base_error = 0.03  # 3% base measurement error
                # Compute reference velocity for normalization
                ref_velocities = analytical_solution(well_positions, nu_e_true, K_true, nu, g, domain_length)
                max_ref_vel = np.max(np.abs(ref_velocities[np.isfinite(ref_velocities)]))
                if max_ref_vel > 0 and np.isfinite(max_ref_vel):
                    flow_dependent_error = 0.02 * abs(u_base) / max_ref_vel
                else:
                    flow_dependent_error = 0.02
                measurement_error = base_error + flow_dependent_error
                
                # Final velocity with all effects
                u_measured = u_base * temp_effect * pressure_effect * well_efficiency
                u_measured += np.random.normal(0, measurement_error)
                
                # Environmental conditions
                temp_measured = reservoir_temp + np.random.normal(0, 5)  # ±5°C variation
                pressure_measured = 20 + 0.01 * reservoir_depth + np.random.normal(0, 1)  # bar
                
                # Store data
                timestamps.append(current_date)
                x_data.append(x_pos / domain_length)  # Normalize
                u_data.append(u_measured)
                errors.append(measurement_error)
                well_ids.append(well_id)
                temperatures.append(temp_measured)
                pressures.append(pressure_measured)
        
        # Campaign metadata
        campaign_data = {
            'x_data': np.array(x_data),
            'u_data': np.array(u_data),
            'errors': np.array(errors),
            'well_ids': np.array(well_ids),
            'timestamps': np.array(timestamps),
            'temperatures': np.array(temperatures),
            'pressures': np.array(pressures),
            'source': 'synthetic_geothermal',
            'n_measurements': len(x_data),
            'campaign_metadata': {
                'campaign_type': 'geothermal_reservoir_characterization',
                'n_wells': n_wells,
                'duration_days': campaign_duration_days,
                'domain_length_m': domain_length,
                'reservoir_depth_m': reservoir_depth,
                'reservoir_temp_c': reservoir_temp,
                'true_parameters': {
                    'nu_e': nu_e_true,
                    'K': K_true,
                    'nu': nu,
                    'g': g
                },
                'environmental_conditions': {
                    'seasonal_variation': True,
                    'pressure_variation': True,
                    'well_efficiency_variation': True
                }
            }
        }
        
        print(f"✅ Generated {len(x_data)} measurements")
        print(f"   • Wells: {n_wells}")
        print(f"   • Duration: {campaign_duration_days} days") 
        print(f"   • True νₑ: {nu_e_true:.4e}")
        print(f"   • True K: {K_true:.4e}")
        
        return campaign_data

def create_sample_data_files():
    """
    Create sample data files in different formats for testing
    """
    print("📁 Creating sample data files for testing...")
    
    # Create data directory
    os.makedirs('sample_data', exist_ok=True)
    
    # Generate sample geothermal data
    generator = RealWorldDataGenerator()
    sample_data = generator.generate_geothermal_campaign(n_wells=8, campaign_duration_days=15)
    
    # Save as CSV
    df_csv = pd.DataFrame({
        'well_id': sample_data['well_ids'],
        'x_position': sample_data['x_data'],
        'velocity': sample_data['u_data'],
        'measurement_error': sample_data['errors'],
        'timestamp': sample_data['timestamps'],
        'temperature': sample_data['temperatures'],
        'pressure': sample_data['pressures']
    })
    df_csv.to_csv('sample_data/geothermal_campaign.csv', index=False)
    
    # Save as JSON
    json_data = {
        'campaign_info': sample_data['campaign_metadata'],
        'wells': []
    }
    
    for well_id in np.unique(sample_data['well_ids']):
        well_mask = sample_data['well_ids'] == well_id
        well_data = {
            'well_id': well_id,
            'position': {'x': sample_data['x_data'][well_mask][0]},
            'measurements': []
        }
        
        for i in np.where(well_mask)[0]:
            measurement = {
                'timestamp': sample_data['timestamps'][i].isoformat(),
                'velocity': float(sample_data['u_data'][i]),
                'error': float(sample_data['errors'][i]),
                'temperature': float(sample_data['temperatures'][i]),
                'pressure': float(sample_data['pressures'][i])
            }
            well_data['measurements'].append(measurement)
        
        json_data['wells'].append(well_data)
    
    with open('sample_data/geothermal_campaign.json', 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    # Save as Excel
    df_csv.to_excel('sample_data/geothermal_campaign.xlsx', index=False)
    
    # Save as pickle
    with open('sample_data/geothermal_campaign.pkl', 'wb') as f:
        pickle.dump(sample_data, f)
    
    print("✅ Sample data files created in 'sample_data/' directory:")
    print("   • geothermal_campaign.csv")
    print("   • geothermal_campaign.json") 
    print("   • geothermal_campaign.xlsx")
    print("   • geothermal_campaign.pkl")
    
    return sample_data

if __name__ == "__main__":
    # Create sample data files for testing
    sample_data = create_sample_data_files()
    
    # Test data loading
    loader = RealWorldDataLoader()
    
    print("\n🧪 Testing data loaders...")
    
    # Test CSV loader
    csv_data = loader.load_csv_data('sample_data/geothermal_campaign.csv')
    print(f"CSV loader: {csv_data['n_measurements']} measurements")
    
    # Test JSON loader
    json_data = loader.load_json_data('sample_data/geothermal_campaign.json')
    print(f"JSON loader: {json_data['n_measurements']} measurements")
    
    # Test universal loader
    excel_data = loader.load_data('sample_data/geothermal_campaign.xlsx')
    print(f"Excel loader: {excel_data['n_measurements']} measurements")
    
    print("\n✅ All data loaders working correctly!")