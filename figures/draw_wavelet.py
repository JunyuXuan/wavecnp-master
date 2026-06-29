import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch


import numpy as np

fruits = ['pi/4', 'pi/3', 'pi/2', '2pi/3', 'pi', '4pi/3','5pi/3',]
sales = [2.0638, 1.8650, 0.8042, 1.6892, 1.7579, 1.7081, 1.9373]

plt.bar(fruits, sales, color='black')
plt.title('Wavelet function impact')
plt.xlabel('Different wavelet functions (e.g., pi/3 is Daubechies; pi/2 is Haar)')
plt.ylabel('Predictive log-likelihood ')
plt.show()