# Description: Functions to analyse timeseries on their spetrum.
# Author:      Tim Sieker
# E-mail:      timsieker@gmail.com


import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import interp1d
from scipy.stats import chi2
import warnings

def check_data(x, t=None, x_threshold=None, fill_value=np.nan):
    '''
    :param x (array_like): timeseries
    :param t (array_like): time, optional
    :param x_threshold (tuple): lower and upper bound for x, values outside of range will be replaced with nans
    :return: x
    '''


    ### Check for nans first
    if np.isnan(x).any():
        warnings.warn('Nans detected in data. Perform interpolation.')

    ### check for missing time steps
    if t is not None:
        t_diff = np.diff(t)
        timesteps = np.unique(t_diff)

        if len(timesteps) != 1:
            warnings.warn('Uneven timesteps detected in data. Perform interpolation on a new time array (t_new).')


    ### check for values outside given range
    if x_threshold is not None:
        lower_bound  = x_threshold[0]
        upper_bound = x_threshold[1]

        mask = (x < lower_bound) | (x > upper_bound)

        if fill_value is not None:
            x[mask] = fill_value

            warnings.warn('Problematic values detected in data. Perform interpolation.')
        else:
            warnings.warn('Problematic values detected in data.')




def dft(x):
    '''
    :param x (array_like): timeseries
    :param t (array_like): time, optional
    :return: Discrete Fourier Transform of x
    '''
    N = len(x)
    t = np.arange(N)

    f = np.arange(N).reshape((N, 1))
    M = np.exp(-2j * np.pi * f * t / N)

    return np.dot(M, x)

def interpolate(x, t=None, t_new=None, kind='linear'):
    '''
    See scipy.interpolate.interp1d
    :param x (array_like): timeseries
    :param t (array_like): time values of x
    :param t_new (array_like): time values to interpolate on
    :param kind (str): interpolation method. See scipy.interpolate.interp1d.
    :return (array_like): x with nans replaced with interpolated values
    '''

    if t is None:
        t = np.arange(x.shape[0])

    if t_new is None:
        t_new = t

    boo = ~np.isnan(x)
    tck = interp1d(t[boo], x[boo], kind=kind, bounds_error=False, fill_value=np.nan)
    x = tck(t_new)

    return t_new, x

def cov(x1, x2, axis=0):
    '''
    :param x1 (array_like):
    :param x2 (array_like):
    :param axis (int):
    :return (flaot): covariance of x1 and x2
    '''
    return np.nansum((x1 - np.nanmean(x1, axis=axis)) * (x2 - np.nanmean(x2, axis=axis)), axis=axis)/(x1.shape[0]-1)

def corr(x1, x2, axis=0):
    '''
    :param x1 (array_like):
    :param x2 (array_like):
    :param axis (int):
    :return (flaot): correlation of x1 and x2
    '''
    return cov(x1, x2, axis=axis) / (np.nanstd(x1, axis=axis) * np.nanstd(x2, axis=axis))

def autocorr(x, nlag=100):
    '''
    :param x (array_like): timeseries
    :param nlag (float): number of lags
    :return (array_like): autocorrelation
    '''
    rhos = np.ones(nlag)
    for lag in np.arange(nlag)[1:]:
        rhos[lag] = corr(x[:-lag], x[lag:])
    return rhos

def autocov(x, nlag=100):
    '''
    :param x (array_like): timeseries
    :param nlag (float): number of lags
    :return (array_like): autocovariance
    '''
    gammas = np.ones(nlag) * np.var(x)
    for lag in np.arange(nlag)[1:]:
        gammas[lag] = cov(x[:-lag], x[lag:])
    return gammas


def spectrum(x, nlag=100, ft_method='fft'):
    '''
    :param x (array_like): timeseries
    :param nlag (float): number of lags
    :param ft_method: Fourier Transform method,
    Note: periodogram works, but requires additional parameters to be consistent with other methods (only used here for testing)
    :return (array_like, array_like): frequencies (1/time step), spectral variance
    '''

    x_xcov = autocov(x, nlag=nlag)

    if ft_method == 'fft':
        x_spect = np.abs(np.fft.fft(x_xcov))
        freq = np.fft.fftfreq(nlag)

    elif ft_method == 'dft':
        x_spect = np.abs(dft(x_xcov))
        freq = np.fft.fftfreq(nlag)

    elif ft_method == 'periodogram':

        freq, x_spect = scipy.signal.periodogram(x_xcov)
        x_spect = x_spect * (nlag-1)

    return freq, x_spect


def white_noise(x, nlag=100):
    '''
    :param x (array_like): timeseries
    :param nlag (float): number of lags
    :return (array_like, array_like): spectrum of the fitted white noise process
    '''
    freq = np.fft.fftfreq(nlag)
    x_spect = np.ones(x.shape) * np.var(x)
    return freq, x_spect


def red_noise(x, nlag=100):
    '''
    :param x (array_like): timeseries
    :param nlag (float): number of lags
    :return (array_like, array_like): spectrum of the fitted red noise process

    '''
    x_xcov = autocov(x, nlag=2)
    freq = np.fft.fftfreq(nlag)

    varl0 = x_xcov[0]
    varl1 = x_xcov[1]
    a1 = varl1/varl0

    x_spect = (1-a1**2) * varl0 / (1+a1**2-2*a1*np.cos(2*np.pi * freq))

    return freq, x_spect


def noise_ci(noise, q):
    '''
    :param noise (array_like): spectrum of the fitted red noise process
    :param q (float): quantile
    :return (array_like, array_like): lower/upper confidence interval of the fitted red noise process
    '''

    df = len(noise) * 2
    chi_val = chi2.isf(q=q / 2, df=df)

    noise_upper_ci = noise * (chi_val/df)
    noise_lower_ci = noise * (1-(chi_val/df-1))
    return noise_lower_ci, noise_upper_ci

