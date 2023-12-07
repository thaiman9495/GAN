import torch
from model import Generator
batch_size = 3
input_length = 7

generator = Generator(input_length)
noise = torch.randint(0, 2, size=(batch_size, input_length)).float()
generated_data = generator(noise)

print(noise)
print(generated_data)
