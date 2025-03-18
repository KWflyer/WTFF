import numpy as np
import scipy.stats

# 时域特征
def mean(args):
    return np.mean(args["sig"], axis=-1)

def std(args):
    return np.std(args["sig"], axis=-1)

def square_root_amplitude(args):
    return np.square(np.mean(np.sqrt(np.abs(args["sig"])), axis=-1))

def absolute_mean_value(args):
    return np.mean(np.abs(args["sig"]), axis=-1)

def skewness(args):
    # return np.mean(np.power(args["sig"], 3))
    return scipy.stats.skew(args["sig"], axis=-1)

def kurtosis(args):
    # return np.mean(np.power(args["sig"], 4))
    return scipy.stats.kurtosis(args["sig"], axis=-1)

def variance(args):
    return np.mean(np.power(args["sig"], 2), axis=-1)

def kurtosis_index(args):
    return kurtosis(args)/np.square(np.sqrt(variance(args)))

def peak_index(args):
    return np.max(np.abs(args["sig"]), axis=-1)/std(args)

def waveform_index(args):
    return std(args)/absolute_mean_value(args)

def pulse_index(args):
    return np.max(np.abs(args["sig"]), axis=-1)/absolute_mean_value(args)

def skewness_index(args):
    return skewness(args)/np.power(np.sqrt(variance(args)), 3)

# 频域特征
def freq_mean_value(args):
    return np.mean(args["Pfft"], axis=-1)

def freq_variance(args):
    return 1 / (args["Pfft"].shape[-1] - 1) * np.sum(np.square(args["Pfft"] - np.mean(args["Pfft"], axis=-1, keepdims=True)), axis=-1)

def freq_skewness(args):
    return 1 / np.power(freq_variance(args), 3/2) * np.mean(np.power(args["Pfft"]-np.mean(args["Pfft"], axis=-1, keepdims=True), 3), axis=-1)

def freq_steepness(args):
    return 1 / np.power(freq_variance(args), 2) * np.mean(np.power(args["Pfft"]-np.mean(args["Pfft"], axis=-1, keepdims=True), 4), axis=-1)

def gravity_frequency(args):
    return np.sum(args["Afft"] * args["Pfft"], axis=-1, keepdims=True)/np.sum(args["Pfft"], axis=-1, keepdims=True)

def freq_standard_deviation(args):
    return np.sqrt(np.sum(np.square(args["Afft"] - gravity_frequency(args)) * args["Pfft"], axis=-1) / np.mean(args["Pfft"], axis=-1))

def freq_root_mean_square(args):
    return np.sqrt(np.sum(np.power(args["Afft"], 2) * args["Pfft"], axis=-1) / np.sum(args["Pfft"], axis=-1))

def average_freq(args):
    return np.sqrt(np.sum(np.power(args["Afft"], 4) * args["Pfft"], axis=-1) / np.sum(np.power(args["Afft"], 2) * args["Pfft"], axis=-1))

def regularity_degree(args):
    return  np.sum(np.power(args["Afft"], 2) * args["Pfft"], axis=-1) / np.sqrt(np.sum(args["Pfft"], axis=-1) / np.sum(np.power(args["Afft"], 4) * args["Pfft"], axis=-1))

def variation_parameter(args):
    return freq_standard_deviation(args) / gravity_frequency(args).reshape(-1)

def eighth_order_moment(args):
    return np.sum(np.power(args["Afft"] - gravity_frequency(args), 3) * args["Pfft"] / args["Pfft"].shape[-1] / np.power(freq_standard_deviation(args).reshape(-1, 1), 3), axis=-1)
    #  / args["Pfft"].shape[-1] / np.power(freq_standard_deviation(args), 3)

def sixteenth_order_moment(args):
    return np.sum(np.power(args["Afft"] - gravity_frequency(args), 4) * args["Pfft"] / args["Pfft"].shape[-1] / np.power(freq_standard_deviation(args).reshape(-1, 1), 4), axis=-1)
