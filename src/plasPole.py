#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Plasmon-pole model fitting for BSE calculations                          #
#    Fits G(iΩ) data to single plasmon-pole models.                           #
#                                                                             #

import numpy as np
import h5py
from scipy.optimize import least_squares

AU2EV = 27.2114  # Hartree to eV conversion factor

def plasmon_model(z, Finf, S, wp):
    """Single plasmon-pole model for F(z)."""
    return Finf + 2 * wp * S / (wp**2 - z**2)


def fit_plasmon_pole(Omega, Fdata, F0=None, Finf=None):
    """
    Fit F(iOmega) data to a single plasmon-pole model:
        F(z) = Finf + S / (wp^2 - z^2).
    """
    z = 1j * Omega

    # Weights: downweight high-frequency points
    # Frequencies near zero imaginaries (iΩ = 0) are more important
    w = 1.0 / (1.0 + Omega**2)
    # Initial guess for wp
    wp0 = 0.001

    def residuals(params):
        if Finf is None:
            Finf_fit, S, wp = params
        else:
            Finf_fit = Finf
            S, wp = params
        model_vals = plasmon_model(z, Finf_fit, S, wp)
        diff = Fdata - model_vals
        # Weighted residuals, stack real+imag parts
        res = np.concatenate([(w * diff.real), (w * diff.imag)])
        # non-weighted residuals, stack real+imag parts
        # res = np.concatenate([(diff.real), (diff.imag)])
        return res

    # Initial guesses
    if F0 is None and Finf is None:
        Finf0 = Fdata[-1].real
        S0 = (Fdata[0].real - Finf0) * 1.0
        x0 = [Finf0, S0, wp0]
        bounds = ([-np.inf, -np.inf, 1e-8], [np.inf, np.inf, np.inf])
        # bounds = ([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])
    else:
        S0 = (F0 - Finf) * 1.0
        x0 = [S0, wp0]
        bounds = ([-np.inf, 1e-8], [np.inf, np.inf])
        # bounds = ([1e-8, -np.inf], [np.inf, np.inf])

    res = least_squares(residuals, x0, bounds=bounds)

    if Finf is None:
        Finf_fit, S_fit, wp_fit = res.x
    else:
        Finf_fit = Finf
        S_fit, wp_fit = res.x

    return {
        "Finf": Finf_fit,
        "S": S_fit,
        "wp": wp_fit,
        "residual_norm": np.linalg.norm(res.fun)
    }


def fit_G_update(Fdata, ir_file, beta=1000):
    """
    Load F(iOmega) data from BSE output file, fit to plasmon-pole model,
    and return fitted parameters.
    """
    with h5py.File(ir_file, 'r') as f:
        wgrid = f["/bose/wsample"][()]
        wgrid = 2 * wgrid * np.pi / beta
    
    print(f"Fdata shape: {Fdata.shape}")
    Fdata = Fdata.real
    n_exc = Fdata.shape[1]
    niw = wgrid.shape[0]

    print("Fitting to single plasmon-pole model.")
    S_data = np.zeros(n_exc, dtype=np.float64)
    wpole_data = np.zeros(n_exc, dtype=np.float64)
    Finf_data = np.zeros(n_exc, dtype=np.float64)
    res_norm_data = np.zeros(n_exc, dtype=np.float64)

    for exc in range(n_exc):
        try:
            Finf = Fdata[-1, exc]
            F0 = Fdata[niw // 2, exc]
            fit_results = fit_plasmon_pole(wgrid, Fdata[:, exc], F0=F0, Finf=Finf)
            
            # Store results
            S_data[exc] = fit_results['S']
            wpole_data[exc] = fit_results['wp']
            Finf_data[exc] = fit_results['Finf']
            res_norm_data[exc] = fit_results['residual_norm']
        except Exception as e:
            print(f"Error fitting excitation {exc}: {e}")

    return wpole_data, S_data, Finf_data, res_norm_data

