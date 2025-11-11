# BSE@scGW

# The pristine form

$$
\chi_{ijkl} = \Pi_{ijkl} + \Pi_{ijqp}\Xi_{pqnm}\chi_{mnkl}
$$

As of now this is solved iteratively. The complexity would be $\mathcal O (N^8_{AO})$.

The approximate form of BSE kernel is:

$$
\Xi_{pqnm} \approx U_{pqmn} - W_{pqnm} 
$$

Note that this is a total kernel. It doesn’t differentiate singlet/triplet/forbidden excitations. 

# The two-point contraction

Contracting the four-point kernel to two-point (real space):

$$
^2\chi(1,2) = \sum_{1^+,2^+}\ ^4\chi(1,1^+,2^+,2)
$$

For the bare Coulomb interaction in real space

$$
U (r_1,r'_1;r_2,r'_2)=U (r_1,r_2) \delta(r_1,r_1') \delta(r_2,r'_2)
$$

Contracting to microscopic dielectric function:

$$
\epsilon_{ij}^{-1}(i\omega) = \delta_{ij} + \sum_k\ U_{ik}\chi_{kj}(i\omega)
$$

Invert microscopic dielectric function to macroscopic dielectric function

$$
\epsilon_{ij}(i\omega) = 1/\epsilon^{-1}_{ij}(i\omega)
$$

Define an auxiliary function 

$$
f(i\omega_n) = \epsilon_{ij}(i\omega) - \delta_{ij}
$$

The spectral function of $f(i\omega_n)$ should be the same as $\epsilon_{ij}(i\omega)$ as the spectral function of a constant ($\delta_{ij}$) shoud be zero.

# The inversion method

Other than iterating over four-point quantity, we can also try to invert this:

$$
\mathcal I - \Xi \Pi
$$

The matrix size would be $N^2_{AO}\times N^2_{AO}$, then complexity would be $\mathcal O (N^6_{AO})$
