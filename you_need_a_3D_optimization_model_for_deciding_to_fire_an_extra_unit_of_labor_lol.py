import numpy as np
import matplotlib.pyplot as plt



def profit1(x1, x2, p, w1, w2):
    pf = p * (x1 + x2)
    return pf - w1 * x1 - w2 * x2

x1_values = np.linspace(0, 100, 100)  
x2_values = np.linspace(0, 100, 100)  

x1, x2 = np.meshgrid(x1_values,x2_values)


p = 100
w1 = 50  
w2 = 50


profit_values = profit1(x1,x2,p,w1,w2)


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1,x2,profit_values,cmap='viridis')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Profit')
ax.set_title('Profit Function 3D Surface Plot')
plt.show()


