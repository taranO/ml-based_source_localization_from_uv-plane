import os
import sys
import numpy as np
import logging as log
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt
import h5py
from datetime import datetime

# ======================================================================================================================
def set_log_config(is_debug=True):
    log_level = log.DEBUG if is_debug else log.INFO
    log.basicConfig(stream=sys.stdout, format='%(levelname)s: %(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=log_level)

def none_or_str(value):
    if value == 'None':
        return None
    return value

def makeDir(dir):
    """Verifies if a directory exists and creates it if does not exist

    Args:
        dir: directory path

    Returns:
        str: directory path """

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1. / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

# --- DFT ---------------------------------
def fftTransform(image):
    spectrum = np.fft.fftshift(np.fft.fft2(image, norm=None))
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)
    real = np.real(spectrum)
    imag = np.imag(spectrum)

    return spectrum, magnitude, phase, real, imag


def ifftTransform(spectrum, magnitude=None, phase=None):
    if spectrum is None:
        spectrum = np.multiply(magnitude, np.exp(1j * phase))

    iimage = np.fft.ifft2(np.fft.ifftshift(spectrum), norm=None)

    return np.real(iimage)

def normalize(image):
    image -= np.min(image)
    if np.max(image) != 0:
        image /= np.max(image)

    return image

def readH5(file, key):
    data = []
    with h5py.File(file, 'r') as F:
        if isinstance(key, str) and key != "":
            data = np.array(F[key])
        elif isinstance(key, list) and len(key):
            for i in range(len(key)):
                data.append(np.array(F[key[i]]))
        else:
            raise ValueError('Undefined keys')

    return data


def normalize_channel_wise(image):
    image -= np.min(image, axis=(1,2), keepdims=True)
    image /= np.max(image, axis=(1,2), keepdims=True)

    return image

def mse_channel_wise(X, Y):
    return np.mean(np.power(X-Y, 2), axis=(1,2))

def calculateSNR(total_flux, beam_size=[], source_size=[], snr_type=1, rms=50):
    if snr_type == 1:
        snr = total_flux * (10 ** 6) / rms
    else:
        if len(beam_size) == 0:
            raise ValueError("Invalid beam size")
        if len(source_size) == 0:
            raise ValueError("Invalid source size")

        snr = ((10 ** 6) / rms) * ((total_flux * beam_size[0] * beam_size[1]) / (
                np.sqrt(beam_size[0] ** 2 + source_size[0] ** 2) * np.sqrt(beam_size[0] ** 2 + source_size[1] ** 2)))

    return snr

def updateSNRcount(SNR, src, snr_ind=2, snr_max=9):
    src = np.asarray(src)
    snrs = getSNR(src[:, snr_ind], snr_max=snr_max)
    for i in snrs:
        SNR[i] += 1

    return SNR, snrs

def getSNR(snr, snr_max=10):
    snr = np.round(snr).astype(int)
    if isinstance(snr, np.int64):
        snr = snr if snr <= snr_max else snr_max
    else:
        snr[snr > snr_max] = snr_max

    return snr

def to_print(message):
    now = datetime.now()
    print(f"{now.strftime('%d.%m.%Y %H:%M:%S')}: {message}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
