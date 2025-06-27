import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

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


    def plot_results(self):
        # Create a high-resolution domain for plotting the final solution
        x_plot = np.linspace(0, self.H, 200)
        u_analytical = analytical_solution(x_plot, nu_e_true, K_true, self.nu, self.g, self.H)
        
        x_plot_torch = torch.tensor(x_plot, dtype=torch.float32, device=device).view(-1, 1)
        self.u_net.eval() # Set model to evaluation mode
        with torch.no_grad():
            u_pred_plot = self.net_u(x_plot_torch).cpu().numpy()

        # Plotting
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        # 1. Solution Plot
        axs[0].plot(x_plot, u_analytical, 'b-', label='Analytical Solution', linewidth=2)
        axs[0].plot(x_plot, u_pred_plot, 'r--', label='gPINN Prediction', linewidth=2)
        axs[0].plot(self.x_data.cpu().numpy(), self.u_data.cpu().numpy(), 'ko', label='Sensor Data', markersize=8)
        axs[0].set_title('Fluid Velocity Comparison')
        axs[0].set_xlabel('Position x')
        axs[0].set_ylabel('Velocity u(x)')
        axs[0].legend()

        # 2. Parameter Convergence Plot
        epochs = len(self.loss_history)
        axs[1].plot(range(epochs), self.nu_e_history, 'g', label=r'Predicted $\nu_e$')
        axs[1].axhline(y=nu_e_true, color='g', linestyle='--', label=r'True $\nu_e$')
        axs[1].plot(range(epochs), self.K_history, 'm', label='Predicted K')
        axs[1].axhline(y=K_true, color='m', linestyle='--', label='True K')
        axs[1].set_title('Parameter Convergence')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Parameter Value')
        axs[1].set_yscale('log')
        axs[1].legend()

        # 3. Loss History Plot
        axs[2].plot(range(epochs), self.loss_history, 'k-')
        axs[2].set_title('Loss History')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('Total Loss')
        axs[2].set_yscale('log')
        
        plt.tight_layout()
        plt.show()


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

    # Visualize the Results
    solver.plot_results()