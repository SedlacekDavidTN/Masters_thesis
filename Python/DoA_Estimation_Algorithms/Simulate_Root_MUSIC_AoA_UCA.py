# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:30:27 2024

@author: sedla
"""

import numpy as np
import scipy

def T(uca_radius_m: float, frequency_Hz: float, N: int) -> np.ndarray:
    x, L = xi(uca_radius_m, frequency_Hz)
    # J
    J = np.diag([1.0 / ((1j**v) * scipy.special.jv(v, x)) for v in range(-L, L + 1, 1)])

    # F
    F = np.array([[np.exp(2.0j * np.pi * (m * n / N)) for n in range(0, N, 1)] for m in range(-L, L + 1, 1)])

    return (J @ F) / float(N)

def xi(uca_radius_m, frequency_Hz):
    wavelength_m = 3*10**8 / frequency_Hz
    x = 2.0 * np.pi * uca_radius_m / wavelength_m
    L = int(np.floor(x))
    return x, L

## Set simulation parameters
sample_rate = 1e6
n = 1000 # Samples
freq = 200e3 # Frequency

n_elem = 5 # Number of array elements
n_signals = 1 # Number of estimated signals
theta_deg = 10 # Angle of incidence in degrees
radius = 0.5 
radius_m = (radius / np.sqrt(2 * (1 - np.cos(2 * np.pi / n_elem)))) # Calculate radius [m]

phi_step = 2 * np.pi / n_elem # Angular distance between elements
elem_step = np.zeros((1,n_elem)) # Angular element position matrix [rad]
for i in range(5):
    elem_step[0,i] = phi_step * i

## Create the signal
t = np.arange(n) / sample_rate # Time vector
signal = np.exp(-2j * np.pi * freq * t) # Signal vector
signal = signal.reshape(1,-1)

## Simulate the array
theta = theta_deg * np.pi / 180
a = np.exp(-2j * np.pi * radius * np.cos(theta-elem_step)) # Calculate the array factor 
a = a.reshape(-1,1)
noise = np.random.randn(n_elem, n) + 1j*np.random.randn(n_elem, n) # Generate noise

## Get signals at array elements
r = a @ signal + 0.1*noise

r = T(radius_m, freq, r.shape[0])
r = np.flip(r)

## Estimate the spatial covariance matrix
R_xx = r @ r.conj().T / n

M = R_xx.shape[0]

## Obtain eigenvectors and their eigenvalues
w,v = np.linalg.eig(R_xx)

# Sort eigenvectors to obtain the noise subsopace
eig_val_order = np.argsort(np.abs(w))
v_sorted = v[:, eig_val_order]

Vn = np.zeros((M,M-n_signals), dtype=np.complex64)
for i in range(M-n_signals):
    Vn[:,i] = v_sorted[:,i]
    
ct = Vn @ Vn.conj().T

## Define the polynomial using the noise subspace
p_coeff = np.zeros(2 * M - 1, dtype=np.complex64)

for i in range(-M + 1, M):
    p_coeff[i + (M - 1)] = np.trace(ct, i)
    
all_roots = np.roots(p_coeff)

candidate_roots_abs = np.abs(all_roots)
sorted_idx = candidate_roots_abs.argsort()[(M - 1 - n_signals) : (M - 1)]

valid_roots = all_roots[sorted_idx]
args = np.angle(valid_roots)

doas = np.arcsin(args / (radius * 2.0 * np.pi))

doas_deg = np.rad2deg(doas)


