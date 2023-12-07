import numpy as np
import matplotlib.pyplot as plt

from utility import create_even_dataset

min_int = 0
max_int = 128
data = np.round(np.random.normal(loc=32, scale=10, size=5000)) * 2

# Clean data
wrong_data = [idx for idx, value in enumerate(data) if value < min_int or value > max_int]
data = np.delete(data, wrong_data)

# Plot:
axis = np.arange(start=min(data), stop=max(data) + 2)
plt.hist(data, bins=axis)
plt.show()
