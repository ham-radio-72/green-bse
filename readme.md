# BSE Solver for Matsubara Green's Function

A common way to solve the BSE is to build an effective Hamiltonian eigenvalue problem, which resembles the equation found in TD-DFT and TD-HF methods. 

$$
\begin{pmatrix}
\textbf{A} & \textbf{B} \\ 
-\textbf{B}^* & -\textbf{A}^* 
\end{pmatrix}\begin{pmatrix}
\textbf{X}_n \\ 
\textbf{Y}_n 
\end{pmatrix} = \omega_n\begin{pmatrix}
\textbf{X}_n \\ 
\textbf{Y}_n 
\end{pmatrix}
$$

On the Bosonic Matsubara frequency grid, we can separate the dynamic part as a perturbation:

$$
A_{ia,jb} = (\Delta \epsilon_{ia,jb} +\kappa U_{ia,jb} - W_{ij,ab}^{\infty} ) +(- W_{ij,ab} + W_{ij,ab}^{\infty} )
$$
$$
B_{ia,jb} = (\kappa U_{ia,bj} - W_{ib,aj}^{\infty} ) + (- W_{ib,aj}+W_{ib,aj}^{\infty} )
$$
$$
\textbf{H}^{\mathrm{eff}}(i\Omega_n) = \textbf{H}^{\infty} + \textbf{H}^{\mathrm{dyn}}(i\Omega_n) =
\begin{pmatrix}
\textbf{A}^{\infty} & \textbf{B}^{\infty} \\ 
-\textbf{B}^{\infty*} & -\textbf{A}^{\infty*} 
\end{pmatrix} + 
\begin{pmatrix}
\tilde{W}^{\mathrm{A}} & \tilde{W}^{\mathrm{B}}\\ 
-\tilde{W}^{\mathrm{B}} & -\tilde{W}^{\mathrm{A}} 
\end{pmatrix} 
$$

The result response function $\textbf{F}$ can be fitted as a single plasmon pole for each excitation.

$$\textbf{F} (z) \approx \textbf{F}_{\infty}+\frac{S}{ z^2- \omega_p^2 }$$

The fitted parameters $\omega$ and $S$ represent excitation energies and intensities. 
