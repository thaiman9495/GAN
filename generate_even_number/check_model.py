import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Generator
from utility import sample_binary_noise, convert_float_matrix_to_int_list

n_generated_data = 1000
max_int = 128

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

input_len = int(math.log(max_int, 2))
leaned_model = torch.load('learned_model.pt')
generator = Generator(input_len).to(device)
generator.load_state_dict(leaned_model)
generator.eval()

data_generated = []
for _ in range(n_generated_data):
    noise = sample_binary_noise(input_len, 10).to(device)
    fake_data = generator(noise).to(device)
    number = convert_float_matrix_to_int_list(fake_data)
    data_generated.append(number)

data_generated = np.concatenate(data_generated, axis=0)
print(data_generated)

# Plot:
axis = np.arange(start=min(data_generated), stop=max(data_generated) + 2)
plt.hist(data_generated, bins=axis)
plt.show()


