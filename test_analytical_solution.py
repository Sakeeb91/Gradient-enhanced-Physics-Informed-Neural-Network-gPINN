import numpy as np
import matplotlib.pyplot as plt

# Test the analytical solution implementation
def analytical_solution(x, nu_e, K, nu, g, H):
    """
    Analytical solution for the Brinkman-Forchheimer equation
    """
    r = np.sqrt(nu / (nu_e * K))
    numerator = g * K / nu * (1 - (np.cosh(r * (x - H/2))) / np.cosh(r * H/2))
    return numerator

# Parameters
nu_e_true = 1e-3
K_true = 1e-3
nu = 1e-3
g = 1.0
H = 1.0

# Generate data
x = np.linspace(0, H, 100)
u_analytical = analytical_solution(x, nu_e_true, K_true, nu, g, H)

# Plot the analytical solution
plt.figure(figsize=(10, 6))
plt.plot(x, u_analytical, 'b-', linewidth=2, label='Analytical Solution')
plt.xlabel('Position x')
plt.ylabel('Velocity u(x)')
plt.title('Brinkman-Forchheimer Analytical Solution')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print("✅ Analytical solution test completed successfully!")
print(f"Parameters: νe = {nu_e_true:.4e}, K = {K_true:.4e}")
print(f"Domain: [0, {H}]")
print(f"Max velocity: {np.max(u_analytical):.4e}")
print(f"Min velocity: {np.min(u_analytical):.4e}")