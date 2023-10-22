import numpy as np
import matplotlib.pyplot as plt



# slope of LM curve depends on the sign of r0
r0 = 1.6
Y0 = 100
G0 = 100
M0 = 100


r_values = np.linspace(r0 - 0.1, r0 + 0.1, 200)

#Y values for IS and LM curves
IS_values = (0.4 * Y0 - 0.2 * r_values - r_values**2 + G0 - 8)
LM_values = (Y0 - 0.6 * r_values**2)


plt.figure(figsize=(8, 6))
plt.plot(r_values, IS_values, label='IS Curve')
plt.plot(r_values, LM_values, label='LM Curve')
plt.xlabel('Interest Rate (r)')
plt.ylabel('National Income (Y)')
plt.title('IS and LM Curves')
plt.grid(True)
plt.legend()
plt.axhline(Y0, color='r', linestyle='--', label='Y0', alpha=0.7)
plt.axvline(r0, color='g', linestyle='--', label='r0', alpha=0.7)
plt.legend()
plt.show()
