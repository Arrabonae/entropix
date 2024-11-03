import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def inverse_log_func(x, a, b, c):
    return a - b * np.log(x + c)

# Generate sample data (you would replace this with your actual data)
np.random.seed(0)
x = np.linspace(0.1, 2.5, 100)
y_sets = [inverse_log_func(x, a, 1, 0.1) + np.random.normal(0, 0.05, 100) 
          for a in np.linspace(0.5, 2.5, 5)]

# Plotting
plt.figure(figsize=(10, 6))
for y in y_sets:
    # Fit the curve
    popt, _ = curve_fit(inverse_log_func, x, y)
    
    # Plot original points
    plt.scatter(x, y, alpha=0.3)
    
    # Plot fitted curve
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_fit = inverse_log_func(x_smooth, *popt)
    plt.plot(x_smooth, y_fit, '-', linewidth=2)

plt.xlabel('Entropy')
plt.ylabel('Varentropy')
plt.title('Entropy vs Varentropy (Logarithmic Regression)')
plt.grid(True)
plt.xlim(0.5, 2.6)
plt.ylim(0, 2.7)
plt.show()

# Print function for one of the curves
popt, _ = curve_fit(inverse_log_func, x, y_sets[0])
print(f"Example function: y = {popt[0]:.2f} - {popt[1]:.2f} * log(x + {popt[2]:.2f})")