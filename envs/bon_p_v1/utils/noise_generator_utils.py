import numpy as np
from scipy.stats import truncnorm


def constant_noisy_data(data, noise):
    if data is None:
        return None
    return data + noise

def uniform_noisy_data(data, lower, upper):
    if data is None:
        return None
    noise = np.random.uniform(lower, upper, size=data.shape)
    return data + noise

def gaussian_noisy_data(data, mean, std):
    if data is None:
        return None
    noise = np.random.normal(mean, std, size=data.shape)
    return data + noise

def truncated_gaussian_noisy_data(data, mean, std, lower, upper):
    if data is None:
        return None
    a = (lower - mean) / std
    b = (upper - mean) / std
    noise = truncnorm.rvs(a, b, loc=mean, scale=std, size=data.shape)
    return data + noise
