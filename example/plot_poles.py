#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Plot the plasmon-pole model fitted results for BSE calculations          #
#                                                                             #

import h5py
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Read arguments.
parser = argparse.ArgumentParser(
    description="Nevanlinna analytic continuation for n-th iteration: Uses the updated Fock"
)
parser.add_argument(
    "--bse_file", type=str, default="bse.h5",
    help="Input of BSE file"
)
parser.add_argument(
    "--w_max", type=float, default=20,
    help="Maximum frequency for plotting (in units of eV)."
)
parser.add_argument(
    "--eta", type=float, default=1e-4,
    help="Broadening factor for plotting (in units of Hartree)."
)
parser.add_argument(
    "--plot_file", type=str, default="plasmon_pole_fit.png",
    help="File name for saving the plot."
)

AU2EV = 27.2114

# Parse arguments
args = parser.parse_args()

bse_path   = args.bse_file
w_max      = args.w_max
eta        = args.eta
plot_file  = args.plot_file

def readBSEh5(bse_path):
    """Read BSE data from HDF5 file."""
    with h5py.File(bse_path, 'r') as f:
        pole_loc = f["/PoleFit/data"][()]
        pole_str = f["/SFit/data"][()]
    return pole_loc, pole_str

def plot_spectral_function(pole_loc, pole_str, eta, wmin=0, wmax=w_max, npts=2000):
    w = np.linspace(wmin, wmax, npts)

    # single-pole Lorentzian spectral function
    A = np.zeros_like(w)
    for i in range(len(pole_loc)):
        if pole_str[i] > 0:
            A += pole_str[i] * eta / ((w - pole_loc[i])**2 + eta**2)
        elif pole_str[i] < 0:
            A += pole_str[i] * eta / ((w + pole_loc[i])**2 + eta**2)

    plt.figure(figsize=(6,3))
    plt.plot(w, A)
    plt.xlabel("ω")
    plt.ylabel("A(ω)")
    plt.xlim(wmin, wmax)
    plt.title("BSE Spectral Function")
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def main():
    """Main entry point."""
    pole_loc, pole_str = readBSEh5(bse_path)
    pole_loc *= AU2EV 
    # print("Pole Locations (eV):", pole_loc)
    # print("Pole Strengths:", pole_str)
    plot_spectral_function(pole_loc, pole_str, eta, wmax=w_max)
    plt.savefig(plot_file, dpi=300)

if __name__ == "__main__":
    main()