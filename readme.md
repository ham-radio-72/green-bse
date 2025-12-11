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

## Main Features

- Solves dynamic BSE equations on imaginary frequency grid
- Provides dynamic solutions and spectral function via plasmon pole fitting
- Draws molecular orbitals for each particle-hole pair
- Quasi-particle (QP) approximation for *GW* energy levels
- Tamm-Dancoff approximation (TDA) option for Casida equation
- Currently only supports molecular or single *k*-point systems

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
export SCRIPTDIR=/your/code/directory/green-bse/example
export INPUTDIR=/your/input/directory
export SIMDIR=/your/input/directory
export IRDIR=/your/irgrid/directory

python $SCRIPTDIR/solveCasida_main.py --type singlet \
       --calc_pi 1 --qpac 1 --monitor 1 \
       --iter -1 --iter_W -1 --beta 1000 \
       --sim $SIMDIR/sim.h5 \
       --int_path $INPUTDIR/df_hf_int/ \
       --input $INPUTDIR/input.h5 \
       --ir_file $IRDIR/1e5_136.h5 \
       --output $SIMDIR/bseCasida.h5 
```

Some parameter definitions:

- `calc_pi`: Calculate polarizability on the fly (default: `True`)
- `qpac`: Use QP for *GW* energy levels (default: `True`)
- `monitor`: Memory and parallelization monitoring (default: `True`)
- `iter` and `iter_W`: The iteration number to read from `sim.h5` (default: `-1` for the lastest iteration)
- `bseCasida.h5`: Output containing excitation energies, eigenvectors, and fitted poles
