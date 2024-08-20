import torch
import scipy.signal as signal
import numpy as np
from sklearn.preprocessing import normalize


def rms(x):
    return np.sqrt(np.mean(x * x))


def pk(x):
    return np.max(np.abs(x))

def avg(x):
    return np.mean(x)


class NpToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).contiguous()


class STFT2D:
    def __init__(self, power=False):
        self.power = power

    def __call__(self, x):
        y = np.abs(
            signal.stft(x, fs=25600, nperseg=384, nfft=384, scaling="spectrum")[2][
                :, :128, :128
            ]
        ).astype("float32")
        if self.power:
            y = 20 * np.log10(y)

        return y
