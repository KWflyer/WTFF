import cv2
import pywt
import numpy as np
import scipy.io
import scipy.fft
from scipy import stats
from tqdm import tqdm
from torchvision.transforms import transforms

from data.sequence_aug import *
import data.feature_indicators as feature_indicators


def time_signal_transforms(args):
    if args.dataaug and args.train_mode not in ["finetune", "evaluate", "wgan", "dcgan"]:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            RandomZero(),
            RandomScale(),
            RandomStretch(),
            AddGaussian(),
            Retype()
        ])
    elif args.snr and args.train_mode not in ["finetune", "evaluate"]:
        data_transforms = transforms.Compose([
            AddWhiteNoise(snr=args.snr),
            Normalize(args.normalize_type),
        ])
    elif args.do and args.train_mode not in ["finetune", "evaluate"]:
        data_transforms = transforms.Compose([
            SignalDropout(p=args.do),
            Normalize(args.normalize_type),
        ])
    elif args.fft:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            Signal_FFT()
        ])
    # elif args.cwt:
    #     data_transforms = transforms.Compose([
    #         Normalize(args.normalize_type),
    #         Signal_CWT(),
    #         # transforms.ToTensor(),
    #         # transforms.Resize((2048, 2048))
    #     ])
    else:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type)
        ])
    return data_transforms

def cwt_transforms():
    return transforms.Compose([
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.5)),
        transforms.Normalize(0,1)
    ])


def time_signal_transforms_to_freq(args):
    if args.dataaug and args.train_mode != "finetune":
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            RandomZero(),
            RandomScale(),
            RandomStretch(),
            AddGaussian(),
            Signal_FFT(),
            Retype()
        ])
    else:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            Signal_FFT(),
            Retype()
        ])
    return data_transforms


class Signal_FFT():
    def __init__(self) -> None:
        self.sampling_rate = 64000
        self.fft_size = 2048

    def __call__(self, seq):
        _, _, Afft, _ = signal_fft(seq, self.sampling_rate, self.fft_size)
        return Afft
        # return Afft[:, :Afft.shape[-1]//2]


def time_signal_transforms_to_cwt(args):
    if args.dataaug and args.train_mode != "finetune":
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            RandomZero(),
            RandomScale(),
            RandomStretch(),
            AddGaussian(),
            Signal_CWT(),
            Retype()
        ])
    else:
        data_transforms = transforms.Compose([
            Normalize(args.normalize_type),
            Signal_CWT(),
            Retype()
        ])
    return data_transforms


class Signal_CWT():
    def __init__(self) -> None:
        self.sampling_rate = 64000
        self.fft_size = 2048

    def __call__(self, seq):
        seq = seq.reshape(-1)
        # 数据采样频率
        fs = self.sampling_rate
        t = np.linspace(0, self.fft_size / fs, self.fft_size, endpoint=False)
        wavename = 'morl'
        totalscal = 256
        fc = pywt.central_frequency(wavename)
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        [cwtmatr, frequencies] = pywt.cwt(seq, scales, wavename, 1.0 / fs)
        # cwtmatr = cv2.resize(cwtmatr, dsize=(cwtmatr.shape[1], cwtmatr.shape[1]))
        return cwtmatr.reshape(1, cwtmatr.shape[0], cwtmatr.shape[1])

class Signals_CWT():
    def __init__(self) -> None:
        self.sampling_rate = 64000
        self.fft_size = 2048

    def __call__(self, seq):
        # 数据采样频率
        fs = self.sampling_rate
        t = np.linspace(0, self.fft_size / fs, self.fft_size, endpoint=False)
        wavename = 'morl'
        totalscal = 256
        fc = pywt.central_frequency(wavename)
        cparam = 2 * fc * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)
        [cwtmatr, frequencies] = pywt.cwt(seq, scales, wavename, 1.0 / fs)
        # cwtmatr = cv2.resize(cwtmatr, dsize=(cwtmatr.shape[1], cwtmatr.shape[1]))
        return cwtmatr

class Resize():
    def __call__(self, icwt):
        return cv2.resize(icwt, dsize=(icwt.shape[1], icwt.shape[1]))


def self_cnn_transforms(args, seq, lab):
    chioces = [Normal(), RandomZero(), RandomIncrease(),
               RandomReduction(), TimeDisorder()]
    transform = Compose([chioces[lab], Normalize(args.normalize_type)])
    return transform(seq)


def signal_fft(signal, sampling_rate, fft_size):
    ysignal = signal

    yfft = scipy.fft.fft(ysignal)
    xfft = scipy.fft.fftfreq(fft_size, 1 / sampling_rate)

    # 幅值
    Afft = 1.0 / fft_size * np.abs(yfft)

    # 功率谱
    Pfft = np.power(Afft, 2) / len(Afft) * sampling_rate
    return xfft, yfft, Afft, Pfft


def signals_fft(signal, sampling_rate, fft_size):
    ysignal = signal[:, :fft_size]

    yfft = scipy.fft.fft(ysignal)
    xfft = scipy.fft.fftfreq(fft_size, 1 / sampling_rate)[:fft_size//2]

    # 幅值
    Afft = 2.0 / fft_size * np.abs(yfft[:, :fft_size//2])

    # 功率谱
    Pfft = np.power(Afft, 2) / Afft.shape[1] * sampling_rate
    return xfft, yfft, Afft, Pfft


def process_one_sig(signal, sampling_rate, indicators):
    # 频域变换
    signal = signal.reshape(-1)
    fft_size = 8192 if len(signal) > 8192 else 2048 # FFT 采样长度

    xfft, yfft, Afft, Pfft = signal_fft(signal, sampling_rate, fft_size)

    args = {
        "sig": signal,
        "Afft": Afft.reshape(-1),
        "Pfft": Pfft.reshape(-1)
    }
    sig_findicators = []
    for indicator in indicators:
        sig_findicators.append(getattr(feature_indicators, indicator)(args)),

    return sig_findicators


def process_signals(signal, sampling_rate, indicators):
    # 频域变换
    if len(signal.shape) > 1:
        signal = signal.reshape(signal.shape[0], -1)
    elif len(signal.shape) == 1:
        signal = signal.reshape(1, -1)
    else:
        raise "signal.shape invalid"
    
    fft_size = 8192 if signal.shape[-1] > 8192 else 2048 # FFT 采样长度

    xfft, yfft, Afft, Pfft = signals_fft(signal, sampling_rate, fft_size)

    args = {
        "sig": signal,
        "Afft": Afft,
        "Pfft": Pfft
    }
    sig_findicators = []
    for indicator in tqdm(indicators):
        sig_findicators.append(getattr(feature_indicators, indicator)(args).reshape(-1,1))

    return np.concatenate(sig_findicators, axis=-1)


# sig (1,2048)
def prior_knowledge(sig):
    indecators = [
        "mean", "std", "square_root_amplitude", "absolute_mean_value",
        "skewness", "kurtosis", "variance", "kurtosis_index",
        "peak_index", "waveform_index", "pulse_index", "skewness_index",
        "freq_mean_value", "freq_variance", "freq_skewness", "freq_steepness",
        "gravity_frequency", "freq_standard_deviation", "freq_root_mean_square",
        "average_freq", "regularity_degree", "variation_parameter",
        "eighth_order_moment", "sixteenth_order_moment",
    ]

    # sig_findicators = []
    # for signal in tqdm(sig):
    #     sig_findicators.append(process_one_sig(signal, 2048, indecators))
    sig_findicators = process_signals(sig, 2048, indecators)
    return sig_findicators
