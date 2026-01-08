# BSE Casida Equation Solver for Matsubara Green's Function Methods

This python module implements a solver for the Bethe-Salpeter Equation (BSE) using the Casida formalism on the sparse sampled Matsubara frequency grid. It is designed to compute optical excitation energies and properties within the BSE@*GW* framework.

The Casida equation is formulate as follows.

$$
\begin{pmatrix}
A & B \\ 
-B^* & -A^* 
\end{pmatrix}\begin{pmatrix}
X_n \\ 
Y_n 
\end{pmatrix} = \omega_n\begin{pmatrix}
X_n \\ 
Y_n
\end{pmatrix}
$$

$$
A_{ia,jb} = \Delta \epsilon_{ia,jb} +\kappa U_{ia,jb} - W_{ij,ab}
$$

$$
\Delta \epsilon_{ia,jb} = (\epsilon_a - \epsilon_i)\delta_{ia,jb} =  (\epsilon_a - \epsilon_i)\delta_{ij}\delta_{ab} 
$$

$$
B_{ia,jb} = \kappa U_{ia,bj} - W_{ib,aj}
$$

Refer to our upcoming paper for more theoretical background. 

## Main Features

- Solves dynamic BSE equations on imaginary frequency grid
- Provides dynamic solutions and spectral function via plasmon pole fitting
- Draws molecular orbitals for each particle-hole pair
- Quasi-particle (QP) approximation for *GW* energy levels
- Tamm-Dancoff approximation (TDA) option for Casida equation
- Currently only supports molecular or single *k*-point systems

Python dependency
-----------

| Package    | Version    |
|------------|------------|
| `h5py`     | 3.15.1     |
| `irbasis`  | 2.2.3      |
| `joblib`   | 1.5.3      |
| `numpy`    | 2.4.0      |
| `pyscf`    | 2.6.2      |
| `scipy`    | 1.16.3     |

Input Files
-----------

- `input.h5`: Mean-field (HF/DFT) reference data
- `sim.h5`: Single-particle Green's function reuslts from scGW iterations
- `df_hf_int/`: Path for electron repulsion integrals (ERI) in auxiliary basis
- IR basis file: Intermediate representation grid information

Usage Example
-------------

Job submission scripts are included in the example folder. The usage of main job script:

```bat
export SCRIPTDIR=/your/code/directory/green-bse/script
export INPUTDIR=/your/input/directory
export SIMDIR=/your/input/directory
export IRDIR=/your/irgrid/directory

python $SCRIPTDIR/solveCasida_main.py --type singlet \
       --calc_pi 1 --qpac 1 --monitor 1 --n_jobs -1 \
       --iter -1 --iter_W -1 --beta 1000 \
       --sim $SIMDIR/sim.h5 \
       --int_path $INPUTDIR/df_hf_int/ \
       --input $INPUTDIR/input.h5 \
       --ir_file $IRDIR/1e5_136.h5 \
       --output $SIMDIR/bse_singlet.h5 
```

Some useful parameter definitions:

- `calc_pi`: Calculate polarizability on the fly (default: `True`)
- `qpac`: Use QP for *GW* energy levels (default: `True`)
- `monitor`: Memory and parallelization monitoring (default: `True`)
- `iter` and `iter_W`: The iteration number to read from `sim.h5` (default: `-1` for the lastest iteration). `iter` is for the Green's function and `iter_W` is for screened Coulomb iteraction. They should be the same, but two separate variables are defined for testing purposes. 
- `n_jobs`: Number of threads to be used. (default: `-1` to use all threads available)
- `output`: Output containing excitation energies, eigenvectors, and fitted poles.


Try it out now!
-------------

We have provided an example of dinitrogen molecule in the `cc-pvdz` basis set. 
It is a small enough system that you can run on your desktop.
In '/example/N2_ccpvdz', you can find all the output you needed from DFT and sc*GW* calculations. 

Simply run:

```bat
sh bse_singlet.sh > bse_singlet.log
```

This submits a BSE calculation and the results will be saved in a `.h5` file. 
In the `.log` file you can also find the printed values of static and dynamic particle-hole excitations.
After the BSE calculation is done, you can run these two scripts:

```bat
sh check_MO_molden.sh 
sh plot_plas_pole.sh 
```

`check_MO_molden.sh` generates the `.molden` file for each particle-hole excitation. 
It contains the MO that hosts the electron originally and the MO it is excited to. 
You can use any software that visualizes `.molden` file to identify the excitation type.

`plot_plas_pole.sh` plots the BSE-computed auxiliary response function and the plasmon-pole fitted results.
