import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from datetime import datetime

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Set device to CPU as requested
device = torch.device('cpu')
print(f"Using device: {device}")

# The Physics Model and Ground Truth
# These are the true, hidden parameters we want the network to discover.
nu_e_true = 1e-3
K_true = 1e-3
nu = 1e-3  # Known fluid property (kinematic viscosity)
g = 1.0      # Known external force (pressure gradient)
H = 1.0      # Domain length [0, H]

# Analytical solution from the paper, used to generate synthetic "sensor" data
def analytical_solution(x, nu_e, K, nu, g, H):
    r = np.sqrt(nu / (nu_e * K))
    numerator = g * K / nu * (1 - (np.cosh(r * (x - H/2))) / np.cosh(r * H/2))
    return numerator

# The Neural Network for Approximating u(x)
# A simple fully-connected network
class PINN_Net(nn.Module):
    def __init__(self, num_layers=4, hidden_size=32):
        super(PINN_Net, self).__init__()
        layers = [nn.Linear(1, hidden_size), nn.Tanh()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.Tanh()])
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# The gPINN Solver Class
class BrinkmanForchheimerSolver:
    def __init__(self, x_data, u_data, x_collocation, g, nu, H):
        self.x_data = torch.tensor(x_data, dtype=torch.float32, device=device).view(-1, 1)
        self.u_data = torch.tensor(u_data, dtype=torch.float32, device=device).view(-1, 1)
        self.x_colloc = torch.tensor(x_collocation, dtype=torch.float32, device=device).view(-1, 1)
        self.x_colloc.requires_grad = True # Crucial for computing derivatives

        self.g = g
        self.nu = nu
        self.H = H

        # The neural network that approximates the solution u(x)
        self.u_net = PINN_Net().to(device)

        # The unknown physical parameters we want to learn.
        # We initialize them with some guess.
        self.nu_e = nn.Parameter(torch.tensor([0.01], device=device))
        self.K = nn.Parameter(torch.tensor([0.01], device=device))

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.u_net.parameters()) + [self.nu_e, self.K],
            lr=1e-3
        )
        
        # History for plotting
        self.loss_history = []
        self.nu_e_history = []
        self.K_history = []
        
        # Loss weights (hyperparameters)
        self.lambda_g = 0.1 # Weight for the gradient loss term

    def net_u(self, x):
        """
        Computes u(x) and enforces hard boundary conditions u(0)=0, u(H)=0.
        This is a common trick in PINNs to make learning easier.
        """
        # The term x * (H - x) is zero at x=0 and x=H
        u = x * (self.H - x) * self.u_net(x)
        return u

    def net_f(self, x):
        """
        Computes the PDE residual f and its gradient df/dx.
        f = -nu_e * u_xx + (nu/K) * u - g
        """
        u = self.net_u(x)

        # First derivative
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

        # PDE residual
        residual = -self.nu_e * u_xx + (self.nu / self.K) * u - self.g
        
        # Gradient of the residual (for gPINN)
        residual_x = torch.autograd.grad(residual, x, grad_outputs=torch.ones_like(residual), create_graph=True)[0]

        return residual, residual_x
    
    def loss_func(self):
        """Computes the total loss."""
        # 1. Data Loss (MSE_u)
        u_pred_data = self.net_u(self.x_data)
        loss_u = torch.mean((self.u_data - u_pred_data)**2)

        # 2. Physics Residual Loss (MSE_f)
        f_pred, f_x_pred = self.net_f(self.x_colloc)
        loss_f = torch.mean(f_pred**2)

        # 3. Gradient of Residual Loss (MSE_g) - The 'g' in gPINN
        loss_g = torch.mean(f_x_pred**2)

        # Total Loss
        total_loss = loss_u + loss_f + self.lambda_g * loss_g
        return total_loss

    def train(self, epochs):
        print("Starting training...")
        start_time = time.time()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            loss = self.loss_func()
            loss.backward()
            self.optimizer.step()

            # Store history
            self.loss_history.append(loss.item())
            self.nu_e_history.append(self.nu_e.item())
            self.K_history.append(self.K.item())

            if (epoch + 1) % 1000 == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Loss: {loss.item():.4e}, "
                    f"νe: {self.nu_e.item():.4e}, "
                    f"K: {self.K.item():.4e}, "
                    f"Time: {elapsed_time:.2f}s"
                )
                start_time = time.time()

        print("Training finished.")
        print(f"True values:      νe = {nu_e_true:.4e}, K = {K_true:.4e}")
        print(f"Predicted values: νe = {self.nu_e.item():.4e}, K = {self.K.item():.4e}")


    def create_visualizations(self, save_plots=True, show_plots=True):
        """
        Create comprehensive visualizations with descriptive captions and save to files
        """
        # Create visualization directory if it doesn't exist
        if save_plots:
            viz_dir = "visualizations"
            os.makedirs(viz_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a high-resolution domain for plotting the final solution
        x_plot = np.linspace(0, self.H, 200)
        u_analytical = analytical_solution(x_plot, nu_e_true, K_true, self.nu, self.g, self.H)
        
        x_plot_torch = torch.tensor(x_plot, dtype=torch.float32, device=device).view(-1, 1)
        self.u_net.eval()
        with torch.no_grad():
            u_pred_plot = self.net_u(x_plot_torch).cpu().numpy()
            
            # Compute physics residuals for visualization
            f_pred, f_x_pred = self.net_f(x_plot_torch)
            f_pred_plot = f_pred.cpu().numpy()
            f_x_pred_plot = f_x_pred.cpu().numpy()

        # Set plotting style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # === PLOT 1: Main Results Summary ===
        fig, axs = plt.subplots(1, 3, figsize=(20, 6))
        
        # Solution comparison
        axs[0].plot(x_plot, u_analytical, 'b-', label='Analytical Solution', linewidth=3, alpha=0.8)
        axs[0].plot(x_plot, u_pred_plot, 'r--', label='gPINN Prediction', linewidth=2)
        axs[0].scatter(self.x_data.cpu().numpy(), self.u_data.cpu().numpy(), 
                      c='black', s=100, marker='o', label='Sensor Data', zorder=5, edgecolors='white', linewidth=2)
        axs[0].set_title('Fluid Velocity Field in Porous Medium\n'
                        f'Inverse Problem: Inferring Permeability (K) and Effective Viscosity (νₑ)', 
                        fontsize=12, fontweight='bold')
        axs[0].set_xlabel('Spatial Position x [dimensionless]', fontsize=11)
        axs[0].set_ylabel('Fluid Velocity u(x) [m/s]', fontsize=11)
        axs[0].legend(fontsize=10)
        axs[0].grid(True, alpha=0.3)
        
        # Parameter convergence
        epochs = len(self.loss_history)
        axs[1].plot(range(epochs), self.nu_e_history, 'g-', label=r'Predicted $\nu_e$ (Effective Viscosity)', linewidth=2)
        axs[1].axhline(y=nu_e_true, color='g', linestyle='--', linewidth=2, 
                      label=f'True $\\nu_e$ = {nu_e_true:.1e}', alpha=0.8)
        axs[1].plot(range(epochs), self.K_history, 'm-', label='Predicted K (Permeability)', linewidth=2)
        axs[1].axhline(y=K_true, color='m', linestyle='--', linewidth=2, 
                      label=f'True K = {K_true:.1e}', alpha=0.8)
        axs[1].set_title('Convergence of Inferred Rock Properties\n'
                        'Learning Hidden Physical Parameters from Sparse Sensor Data', 
                        fontsize=12, fontweight='bold')
        axs[1].set_xlabel('Training Epoch', fontsize=11)
        axs[1].set_ylabel('Parameter Value [SI units]', fontsize=11)
        axs[1].set_yscale('log')
        axs[1].legend(fontsize=9)
        axs[1].grid(True, alpha=0.3)
        
        # Loss evolution
        axs[2].plot(range(epochs), self.loss_history, 'k-', linewidth=2)
        axs[2].set_title('Training Loss Evolution\n'
                        'Physics-Informed Learning with Gradient Enhancement', 
                        fontsize=12, fontweight='bold')
        axs[2].set_xlabel('Training Epoch', fontsize=11)
        axs[2].set_ylabel('Total Loss (Data + Physics + Gradients)', fontsize=11)
        axs[2].set_yscale('log')
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{viz_dir}/01_main_results_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{viz_dir}/01_main_results_{timestamp}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # === PLOT 2: Physics Residuals Analysis ===
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # PDE Residual
        axs[0,0].plot(x_plot, f_pred_plot, 'r-', linewidth=2)
        axs[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axs[0,0].set_title('Physics Residual: Brinkman-Forchheimer PDE\n'
                          f'f = -νₑ∇²u + (ν/K)u - g ≈ 0 everywhere', 
                          fontsize=12, fontweight='bold')
        axs[0,0].set_xlabel('Position x', fontsize=11)
        axs[0,0].set_ylabel('Residual f(x)', fontsize=11)
        axs[0,0].grid(True, alpha=0.3)
        
        # Gradient of residual (gPINN enhancement)
        axs[0,1].plot(x_plot, f_x_pred_plot, 'orange', linewidth=2)
        axs[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axs[0,1].set_title('Gradient Enhancement: ∇f ≈ 0\n'
                          'Additional constraint for smoother solutions', 
                          fontsize=12, fontweight='bold')
        axs[0,1].set_xlabel('Position x', fontsize=11)
        axs[0,1].set_ylabel('Residual Gradient ∂f/∂x', fontsize=11)
        axs[0,1].grid(True, alpha=0.3)
        
        # Error analysis
        error = np.abs(u_pred_plot.flatten() - u_analytical)
        axs[1,0].plot(x_plot, error, 'purple', linewidth=2)
        axs[1,0].set_title('Pointwise Absolute Error\n'
                          f'Mean Absolute Error: {np.mean(error):.2e}', 
                          fontsize=12, fontweight='bold')
        axs[1,0].set_xlabel('Position x', fontsize=11)
        axs[1,0].set_ylabel('|u_pred - u_analytical|', fontsize=11)
        axs[1,0].grid(True, alpha=0.3)
        
        # Parameter estimation accuracy
        nu_e_error = abs(self.nu_e.item() - nu_e_true) / nu_e_true * 100
        K_error = abs(self.K.item() - K_true) / K_true * 100
        
        params = ['Effective Viscosity\n(νₑ)', 'Permeability\n(K)']
        errors = [nu_e_error, K_error]
        colors = ['green', 'magenta']
        
        bars = axs[1,1].bar(params, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axs[1,1].set_title('Parameter Estimation Accuracy\n'
                          'Percentage Error in Inferred Rock Properties', 
                          fontsize=12, fontweight='bold')
        axs[1,1].set_ylabel('Relative Error [%]', fontsize=11)
        axs[1,1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, error in zip(bars, errors):
            height = bar.get_height()
            axs[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{error:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{viz_dir}/02_physics_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{viz_dir}/02_physics_analysis_{timestamp}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # === PLOT 3: Training Dynamics ===
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        
        # Parameter evolution over time
        axs[0,0].plot(range(epochs), self.nu_e_history, 'g-', linewidth=2, label='νₑ Evolution')
        axs[0,0].axhline(y=nu_e_true, color='g', linestyle='--', alpha=0.8, linewidth=2)
        axs[0,0].set_title('Effective Viscosity Learning Curve\n'
                          'Convergence to True Value During Training', 
                          fontsize=12, fontweight='bold')
        axs[0,0].set_xlabel('Training Epoch', fontsize=11)
        axs[0,0].set_ylabel('νₑ [Pa·s]', fontsize=11)
        axs[0,0].set_yscale('log')
        axs[0,0].grid(True, alpha=0.3)
        
        axs[0,1].plot(range(epochs), self.K_history, 'm-', linewidth=2, label='K Evolution')
        axs[0,1].axhline(y=K_true, color='m', linestyle='--', alpha=0.8, linewidth=2)
        axs[0,1].set_title('Permeability Learning Curve\n'
                          'Critical Parameter for Geothermal Reservoir Assessment', 
                          fontsize=12, fontweight='bold')
        axs[0,1].set_xlabel('Training Epoch', fontsize=11)
        axs[0,1].set_ylabel('K [m²]', fontsize=11)
        axs[0,1].set_yscale('log')
        axs[0,1].grid(True, alpha=0.3)
        
        # Loss components (if available)
        axs[1,0].plot(range(epochs), self.loss_history, 'k-', linewidth=2)
        axs[1,0].set_title('Total Loss Function Minimization\n'
                          'Combined Data Fitting + Physics Enforcement', 
                          fontsize=12, fontweight='bold')
        axs[1,0].set_xlabel('Training Epoch', fontsize=11)
        axs[1,0].set_ylabel('Loss Value', fontsize=11)
        axs[1,0].set_yscale('log')
        axs[1,0].grid(True, alpha=0.3)
        
        # Final comparison table
        axs[1,1].axis('off')
        table_data = [
            ['Parameter', 'True Value', 'Predicted', 'Error [%]'],
            ['νₑ [Pa·s]', f'{nu_e_true:.2e}', f'{self.nu_e.item():.2e}', f'{nu_e_error:.2f}%'],
            ['K [m²]', f'{K_true:.2e}', f'{self.K.item():.2e}', f'{K_error:.2f}%'],
            ['MAE', '-', f'{np.mean(error):.2e}', '-']
        ]
        
        table = axs[1,1].table(cellText=table_data[1:], colLabels=table_data[0],
                              cellLoc='center', loc='center', bbox=[0, 0.3, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j)] if i > 0 else table[(0, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        
        axs[1,1].set_title('Quantitative Results Summary\n'
                          'gPINN Performance Metrics', 
                          fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f"{viz_dir}/03_training_dynamics_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.savefig(f"{viz_dir}/03_training_dynamics_{timestamp}.pdf", bbox_inches='tight')
        if show_plots:
            plt.show()
        else:
            plt.close()
            
        # Print summary
        if save_plots:
            print(f"\n📊 Visualizations saved to '{viz_dir}/' directory:")
            print(f"   • 01_main_results_{timestamp}.png/pdf")
            print(f"   • 02_physics_analysis_{timestamp}.png/pdf") 
            print(f"   • 03_training_dynamics_{timestamp}.png/pdf")
            
        print(f"\n🎯 Final Results Summary:")
        print(f"   • Effective Viscosity: {self.nu_e.item():.4e} (true: {nu_e_true:.4e}, error: {nu_e_error:.2f}%)")
        print(f"   • Permeability: {self.K.item():.4e} (true: {K_true:.4e}, error: {K_error:.2f}%)")
        print(f"   • Mean Absolute Error: {np.mean(error):.4e}")
        
    def plot_results(self):
        """Legacy function for backward compatibility"""
        self.create_visualizations(save_plots=False, show_plots=True)


# Main Execution
if __name__ == '__main__':
    # Generate Training Data
    # We only have a few "sensor" measurements.
    num_data_points = 5
    x_data = np.linspace(0.1 * H, 0.9 * H, num_data_points)
    u_data = analytical_solution(x_data, nu_e_true, K_true, nu, g, H)
    
    # Add some noise to make it more realistic
    noise_level = 0.01
    u_data += noise_level * np.std(u_data) * np.random.randn(*u_data.shape)

    # Collocation points for enforcing the PDE
    num_collocation_points = 100
    x_collocation = np.linspace(0, H, num_collocation_points)

    # Create and Train the Solver
    solver = BrinkmanForchheimerSolver(
        x_data=x_data,
        u_data=u_data,
        x_collocation=x_collocation,
        g=g, nu=nu, H=H
    )
    
    solver.train(epochs=20000)

    # Create comprehensive visualizations with automatic saving
    print("\n🎨 Generating comprehensive visualizations...")
    solver.create_visualizations(save_plots=True, show_plots=False)
    
    print("\n✅ Training and visualization complete!")
    print(f"📁 Check the 'visualizations/' directory for saved plots.")