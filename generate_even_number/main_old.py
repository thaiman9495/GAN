import math
import torch
import torch.nn as nn

from torch.optim import Adam
from model import Generator, Discriminator
from utility import sample_binary_noise, generate_even_data, convert_float_matrix_to_int_list


n_training_steps = 500
max_int = 128
batch_size = 32
n_steps_discriminator = 2
lr = 0.001
log_freq = 10

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

input_len = int(math.log(max_int, 2))

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

for i in range(n_training_steps):
    # ------------------------------------------------------------------------------------------------------------------
    # Train discriminator
    # ------------------------------------------------------------------------------------------------------------------
    for _ in range(n_steps_discriminator):
        # Generate fake data
        noise = sample_binary_noise(input_len, batch_size).to(device)
        fake_data = generator(noise).to(device).detach()

        # Sample true data
        true_data, true_label = generate_even_data(max_int, batch_size)
        true_data = torch.FloatTensor(true_data).to(device)
        true_label = torch.unsqueeze(torch.FloatTensor(true_label), dim=1).to(device)

        discriminator_out_true = discriminator(true_data)
        discriminator_out_fake = discriminator(fake_data)

        loss_discriminator_true = loss(discriminator_out_true, true_label)
        loss_discriminator_fake = loss(discriminator_out_fake, fake_label)

        loss_discriminator = loss_discriminator_true + loss_discriminator_fake

        optimizer_discriminator.zero_grad()
        loss_discriminator.backward()
        optimizer_discriminator.step()

    # ------------------------------------------------------------------------------------------------------------------
    # Train generator
    # ------------------------------------------------------------------------------------------------------------------

    # Generate fake data
    noise = sample_binary_noise(input_len, batch_size).to(device)
    fake_data = generator(noise).to(device)

    discriminator_out = discriminator(fake_data)

    loss_generator = - loss(discriminator_out, fake_label)

    optimizer_generator.zero_grad()
    loss_generator.backward()
    optimizer_generator.step()

    if i % log_freq == 0:
        # print(fake_data.detach())
        print(f'Step {i} --> {convert_float_matrix_to_int_list(fake_data)}')


