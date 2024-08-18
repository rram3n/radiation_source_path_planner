import numpy as np
import matplotlib.pyplot as plt
import math

class GaussianPlume:
    def __init__(self, Q, u, sigma_x, sigma_y, x0, y0):
        self.Q = Q          # Source strength (e.g., Bq for radiation)
        self.u = u          # Wind speed (m/s)
        self.sigma_x = sigma_x  # Dispersion coefficient in the x-direction
        self.sigma_y = sigma_y  # Dispersion coefficient in the y-direction
        self.x0 = x0        # x-coordinate of the source
        self.y0 = y0        # y-coordinate of the source

    def concentration(self, x, y):
        distance = np.sqrt((x - self.x0)**2 + (y - self.y0)**2)
        if distance == 0:
            return float('inf')  # Avoid division by zero at the source point
        
        exp_part = np.exp(-((x - self.x0)**2) / (2 * self.sigma_x**2) - ((y - self.y0)**2) / (2 * self.sigma_y**2))
        return (self.Q / (2 * math.pi * self.u * self.sigma_y * self.sigma_x)) * exp_part

# Initialize the Gaussian plume instance
plume = GaussianPlume(Q=1e6, u=1, sigma_x=150, sigma_y=150, x0=100, y0=75)

# Create a grid of points
x_range = np.linspace(0, 200, 200)
y_range = np.linspace(0, 200, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = np.zeros_like(X)

# Calculate concentration at each point
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = plume.concentration(X[i, j], Y[i, j])

# Handle infinities by setting them to a large number for visualization
Z[np.isinf(Z)] = 1e5

# Plotting
plt.figure(figsize=(10, 8))
contour = plt.contourf(X, Y, Z, levels=100, cmap='viridis')
plt.colorbar(contour)
plt.plot(100, 75, 'ro')  # Mark the source location
plt.title('Gaussian Plume Concentration Distribution')
plt.xlabel('X Distance (m)')
plt.ylabel('Y Distance (m)')
plt.show()
