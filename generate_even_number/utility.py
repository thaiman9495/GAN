import math
import torch
import numpy as np


def conver_int_to_binary_list(number: int):
    """
    This function aims to create a list of binary representing the not negative input number

    Args:
        number: a positive integer

    Returns: The binary form of the input
    """

    # Check if input is integer and not negative
    if number < 0 or type(number) is not int:
        raise ValueError('Only not negative numbers are allowed')

    return [int(x) for x in list(bin(number)[2:])]


def convert_binary_list_to_int(binary_list: list):
    """
    This function aims to convert an input list of binary to a corresponding integer number
    Args:
        binary_list: input binary list

    Returns: integer

    """
    return int(''.join(str(x) for x in binary_list), base=2)


def generate_even_data(max_int: int, batch_size: int = 16):
    """
    This function aims to generate a batch of even number represting under binary form

    Args:
        max_int: the maximum integer range for generating data
        batch_size: number of even numbers being generated

    Returns: data and label
    """

    # Get the number of binary places needed to represent the maximum number
    max_len = int(math.log(max_int, 2))

    # Sample batch_size number of integers in range from 0 to max_int
    sampled_integers = np.random.randint(0, int(max_int / 2), batch_size)

    # Create a list of labels all ones because all numbers are even
    label = [1] * batch_size

    # Generate a list of binary numbers for training.
    data = [conver_int_to_binary_list(int(x * 2)) for x in sampled_integers]
    data = [([0] * (max_len - len(x))) + x for x in data]

    return data, label


def sample_binary_noise(input_len, batch_size):
    return torch.randint(0, 2, size=(batch_size, input_len)).float()


def convert_float_matrix_to_int_list(float_matrix: np.array, threshold: float = 0.5):
    """Converts generated output in binary list form to a list of integers

    Args:
        float_matrix: A matrix of values between 0 and 1 which we want to threshold and convert to
            integers
        threshold: The cutoff value for 0 and 1 thresholding.

    Returns:
        A list of integers.
    """
    return [int("".join([str(int(y)) for y in x]), 2) for x in float_matrix >= threshold]


def create_even_dataset(max_int: int, n_samples: int):
    data = np.round(np.random.normal(loc=int(max_int/4.0), scale=10, size=n_samples)) * 2

    # Clean data
    wrong_data = [idx for idx, value in enumerate(data) if value < 0 or value > (max_int-1)]

    data_int = np.delete(data, wrong_data)

    max_len = int(math.log(max_int, 2))
    data_bin = [conver_int_to_binary_list(int(x)) for x in data_int]
    data_bin = [([0] * (max_len - len(x))) + x for x in data_bin]

    return data_int, data_bin
