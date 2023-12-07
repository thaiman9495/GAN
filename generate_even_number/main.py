import math

import numpy as np
import torch
import torch.nn as nn

from random import sample
from torch.optim import Adam
from model import Generator, Discriminator
from torch.utils.data import DataLoader
from utility import sample_binary_noise, convert_float_matrix_to_int_list
from utility import create_even_dataset
from dataset import MyDataset


n_epoches = 200
max_int = 128
batch_size = 32
n_steps_discriminator = 4
lr = 0.0001
log_freq = 10

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

input_len = int(math.log(max_int, 2))

# Read data set
data_int, data_bin = create_even_dataset(max_int, 5000)
dataset = MyDataset(data_bin)
data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

# Adversarial network
generator = Generator(input_len).to(device)
discriminator = Discriminator(input_len).to(device)

# Optimizers
optimizer_generator = Adam(params=generator.parameters(), lr=lr)
optimizer_discriminator = Adam(params=discriminator.parameters(), lr=lr)

# Loss function
loss = nn.BCELoss()

# Main training loop
fake_label = torch.zeros(size=(batch_size, 1)).to(device)
true_label = torch.ones(size=(batch_size, 1)).to(device)

for epoch in range(n_epoches):
    loss_generator_log = []
    loss_discriminator_log = []
    for true_data in data_loader:
        # --------------------------------------------------------------------------------------------------------------
        # Train discriminator
        # --------------------------------------------------------------------------------------------------------------
        # Generate fake data
        noise = sample_binary_noise(input_len, batch_size).to(device)
        fake_data = generator(noise).to(device).detach()

        discriminator_out_true = discriminator(true_data.to(device))
        discriminator_out_fake = discriminator(fake_data)

        loss_discriminator_true = loss(discriminator_out_true, true_label)
        loss_discriminator_fake = loss(discriminator_out_fake, fake_label)

        loss_discriminator = loss_discriminator_true + loss_discriminator_fake

        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

        loss_discriminator_log.append(loss_discriminator)

        # --------------------------------------------------------------------------------------------------------------
        # Train generator
        # --------------------------------------------------------------------------------------------------------------

        # Generate fake data
        noise = sample_binary_noise(input_len, batch_size).to(device)
        fake_data = generator(noise).to(device)

        discriminator_out = discriminator(fake_data)

        loss_generator = - loss(discriminator_out, fake_label)

        optimizer_generator.zero_grad()
        loss_generator.backward()
        optimizer_generator.step()

        loss_generator_log.append(loss_generator)

    mean_loss_g = torch.mean(torch.FloatTensor(loss_generator_log))
    mean_loss_d = torch.mean(torch.FloatTensor(loss_discriminator_log))
    print(f'generator loss: {mean_loss_g: .3f}, discriminator loss: {mean_loss_d: .3f}')

    # if epoch % log_freq == 0:
    #     noise = sample_binary_noise(input_len, batch_size).to(device)
    #     fake_data = generator(noise).to(device).detach()
    #     print(f'Epoch {epoch} --> {convert_float_matrix_to_int_list(fake_data)}')

torch.save(generator.state_dict(), 'learned_model.pt')


# for i in range(n_training_steps):
#     # ------------------------------------------------------------------------------------------------------------------
#     # Train discriminator
#     # ------------------------------------------------------------------------------------------------------------------
#     for _ in range(n_steps_discriminator):
#         # Generate fake data
#         noise = sample_binary_noise(input_len, batch_size).to(device)
#         fake_data = generator(noise).to(device).detach()
#
#         # Sample true data
#         true_data = sample(data, batch_size)
#         true_data = torch.FloatTensor(true_data).to(device)
#         # print(convert_float_matrix_to_int_list(true_data))
#
#         discriminator_out_true = discriminator(true_data)
#         discriminator_out_fake = discriminator(fake_data)
#
#         loss_discriminator_true = loss(discriminator_out_true, true_label)
#         loss_discriminator_fake = loss(discriminator_out_fake, fake_label)
#
#         loss_discriminator = loss_discriminator_true + loss_discriminator_fake
#
#         optimizer_discriminator.zero_grad()
#         loss_discriminator.backward()
#         optimizer_discriminator.step()
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # Train generator
#     # ------------------------------------------------------------------------------------------------------------------
#
#     # Generate fake data
#     noise = sample_binary_noise(input_len, batch_size).to(device)
#     fake_data = generator(noise).to(device)
#
#     discriminator_out = discriminator(fake_data)
#
#     loss_generator = - loss(discriminator_out, fake_label)
#
#     optimizer_generator.zero_grad()
#     loss_generator.backward()
#     optimizer_generator.step()
#
#     if i % log_freq == 0:
#         # print(fake_data.detach())
#         print(f'Step {i} --> {convert_float_matrix_to_int_list(fake_data)}')
#
#     torch.save(generator.state_dict(), 'learned_model.pt')



