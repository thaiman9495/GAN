import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from dataset import MyDataset
from utility import create_even_dataset
from utility import convert_float_matrix_to_int_list

data_int, data_bin = create_even_dataset(max_int=128, n_samples=10000)

# Plot:
axis = np.arange(start=min(data_int), stop=max(data_int) + 2)
plt.hist(data_int, bins=axis)
plt.show()

dataset = MyDataset(data_bin)
data_loader = DataLoader(dataset, batch_size=16)

for epoch in range(1):
    for data in data_loader:
        print(convert_float_matrix_to_int_list(data))






