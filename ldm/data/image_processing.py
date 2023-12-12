import numpy as np
import torch


def is_float(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.floating)
    elif torch.is_tensor(x):
        return torch.is_floating_point(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_integer(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.integer)
    elif torch.is_tensor(x):
        return not torch.is_floating_point(x) and not torch.is_complex(x)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def is_between_minus1_1(x):
    correct_range = x.min() >= -1 and x.min() < 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_1(x):
    correct_range = x.min() >= 0 and x.max() <= 1
    if correct_range:
        assert is_float(x)
        return True
    else:
        return False


def is_between_0_255(x):
    correct_range = x.min() >= 0 and x.max() <= 255
    if correct_range:
        assert is_integer(x)
        return True
    else:
        return False


def convert_to_integer(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.uint8)
    elif torch.is_tensor(x):
        return x.to(torch.uint8)
    else:
        raise ValueError('The input is not a numpy array or a torch tensor.')


def convert_to_0_1(x):
    if is_between_minus1_1(x):
        return (x + 1) / 2
    elif is_between_0_1(x):
        return x
    elif is_between_0_255(x):
        return x / 255
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')


def convert_to_0_255(x):
    if is_between_minus1_1(x):
        x = ((x + 1) / 2 * 255)
        return convert_to_integer(x)
    elif is_between_0_1(x):
        x = x * 255
        return convert_to_integer(x)
    elif is_between_0_255(x):
        return x
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')


def convert_to_minus1_1(x):
    if is_between_minus1_1(x):
        return x
    elif is_between_0_1(x):
        return x * 2 - 1
    elif is_between_0_255(x):
        return x / 255 * 2 - 1
    else:
        raise ValueError('The input is not in the range of [0, 1] or [-1, 1] or [0, 255].')