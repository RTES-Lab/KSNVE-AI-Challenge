import torch
import scipy.signal as signal
import numpy as np
import emd
from sklearn.preprocessing import normalize


def rms(x):
    return np.sqrt(np.mean(x * x))


def pk(x):
    return np.max(np.abs(x))


def moving_filter(data, func=np.mean, window_length=32, shift_size=32):
    data = np.asarray(data)
    filtered_data = []

    for start in range(0, len(data) - window_length + 1, shift_size):
        window = data[start : start + window_length]
        window_average = func(window)
        filtered_data.append(window_average)

    return np.array(filtered_data)


class ChannelLast:
    def __call__(self, x):
        return torch.permute(x, (1, 0)).contiguous()


class NpToTensor:
    def __call__(self, x):
        return torch.from_numpy(x).contiguous()


class Decimate:
    def __call__(self, x):
        return signal.decimate(x, 64).astype("float32")


class FFT:
    def __call__(self, x):
        y = np.fft.fftshift(np.fft.fft(x)) / x.shape[1]
        amp = np.abs(y)
        amp = 20 * np.log10(amp)
        # phase = np.angle(y)
        # ret = np.vstack((amp, phase)).astype("float32")
        ret = amp.astype("float32")

        return ret


class MovingFilter:
    def __call__(self, x):
        px = x[0]
        py = x[1]
        u_x = moving_filter(px)
        u_y = moving_filter(py)
        rms_x = moving_filter(px, func=rms)
        rms_y = moving_filter(py, func=rms)
        pk_x = moving_filter(px, func=pk)
        pk_y = moving_filter(py, func=pk)
        ret = np.vstack((u_x, u_y, rms_x, rms_y, pk_x, pk_y)).astype("float32")

        return ret


class Polar:
    def __init__(self, idx=(0, 1, 2, 3)):
        self.idx = idx

    def __call__(self, x):
        px = x[0]
        py = x[1]
        r = np.sqrt(px**2, py**2)
        theta = np.arctan2(py, px)
        ret = np.vstack((px, py, r, theta)).astype("float32")
        ret = ret[self.idx, :]
        if ret.ndim == 1:
            ret = np.expand_dims(ret, 0)

        return ret


class EMD:
    def __call__(self, x):
        y = emd.sift.sift(x[0], max_imfs=4).T
        y = y.astype("float32")
        z = emd.sift.sift(x[1], max_imfs=4).T
        z = z.astype("float32")
        ret = np.vstack((y, z))
        return ret


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


class STFT:
    def __call__(self, x):
        return np.abs(
            signal.stft(x, nperseg=8192)[2][:, :4096, :].reshape((16, 4096))
        ).astype("float32")


class Normalize:
    def __init__(self, norm):
        self.norm = norm
        if self.norm not in ["l1", "l2", "max"]:
            self.norm = "l2"

    def __call__(self, x):
        return normalize(x, norm=self.norm).astype(np.float32)


class Scale:
    def __init__(self, scale):
        self.scale = scale

        if self.scale < 0:
            raise ValueError("Scale factor must be > 0")

    def __call__(self, x):
        return x / self.scale
