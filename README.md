

Data Science toolkit
==============================
Raymond Ji's personal toolkit for data analysis, data wrangling and machine learning.
This repository has been created as preparation for the 2020 Citadel West Coast Data Open.

import numpy as np

def generate_time_series(n, autocorr):
    # Step 1: Generate white noise
    white_noise = np.random.randn(n)
    
    # Step 2: Create desired autocorrelation structure
    # Compute the Fourier transform of the white noise
    spectrum = np.fft.fft(white_noise)
    
    # Create a sequence of complex numbers with magnitude 1 and phase given by the desired autocorrelation
    target_spectrum = np.exp(1j * np.angle(spectrum))
    
    # Multiply the white noise spectrum by the target spectrum
    modified_spectrum = np.sqrt(autocorr) * target_spectrum
    
    # Compute the inverse Fourier transform to get the time series
    ts = np.fft.ifft(modified_spectrum).real
    
    # Step 3: Normalize the series to have mean 0 and standard deviation 1
    ts = (ts - np.mean(ts)) / np.std(ts)
    
    return ts

# Example usage:
n = 1000
autocorr = 0.5
ts = generate_time_series(n, autocorr)
