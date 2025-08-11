"""
Solve for quasiparticle energies. 
Adapted from scgw-analysis.
"""
import numpy as np
import contract as ct
import h5py
from pyscf.gw.gw_ac import AC_pade_thiele_diag, pade_thiele
from scipy.optimize import newton


# Get the diagonal of Self-energy for analytic continuation
def padeSigma(Sigma_tk_int,fock_eigs,beta,mu,tau_h5):
    Sigma_tk_diag = np.einsum('tskii -> tski', Sigma_tk_int)
    Sigma_iw_diag = ct.tau2omegaFTforG(Sigma_tk_diag,beta,tau_h5)
    # get iwsample from tau_h5.
    nao = Sigma_tk_diag.shape[-1]
    ns  = Sigma_tk_diag.shape[1]
    nk  = Sigma_tk_diag.shape[2]
    f = h5py.File(tau_h5, 'r')
    iwsample = f["/fermi/wsample"][()]
    f.close()
    nw = iwsample.shape[0]
    iwsample_pos = iwsample[iwsample > 0]
    iw_pos_for_pade = np.zeros((nao, len(iwsample_pos)))
    for i in range(nao):
        iw_pos_for_pade[i, :] = iwsample_pos[:]
    Sigma_iw_positive = Sigma_iw_diag[iwsample > 0]
    
    nskip = 1
    if nw//2 < 40:
        idx = np.arange(0, nw//2-1, 1)
    else:
        idx1 = np.arange(0, 20, nskip)
        idx2 = np.arange(idx1[-1] + nskip, nw//2-1, 1)
        idx = np.concatenate((idx1, idx2))
        
    print(idx)
    print(iw_pos_for_pade.shape)
    print(Sigma_iw_positive.shape)
    
    iw_inp = iw_pos_for_pade[:,idx]
    Sigma_iw_inp = Sigma_iw_positive[idx]
    
    print("Pade interpolation for Sigma")
    sig_for_pade = np.einsum('wska -> aw', Sigma_iw_inp)
    print(iw_inp.shape)
    print(sig_for_pade.shape)
    
    coeff_a, omega_fit = AC_pade_thiele_diag(sig_for_pade, 1j * iw_inp)
    
    pade_coeff = np.asarray(coeff_a)
    omega_fit = omega_fit.reshape((1,) + omega_fit.shape)
    pade_coeff = pade_coeff.reshape((1,1,) + pade_coeff.shape)
    #omega_fit = omega_fit.reshape((1,1,) + omega_fit.shape)

    print(sig_for_pade[0,0])
    print(iw_pos_for_pade.shape)
    print(nw)
    
    sc_qp_eigs = fock_eigs.copy()
    
    def quasiparticle(omega, s, k, p):

        sigmaR = pade_thiele(
            omega - mu , omega_fit[s, p], pade_coeff[s, k, :, p]
        ).real
        qp_energy = fock_eigs[s, k, p].real + sigmaR

        return omega - qp_energy    
    
    for sp_idx in range(ns):
        for k_idx in range(nk):
            for mo_idx in range(nao):
                try:
                    e = newton(
                        lambda w: quasiparticle(w, sp_idx, k_idx, mo_idx),
                        fock_eigs[sp_idx, k_idx, mo_idx].real,
                        tol=1e-6,
                        maxiter=100,
                        full_output=False
                    )
                    # print(au2ev*e)
                    sc_qp_eigs[sp_idx, k_idx, mo_idx] = e
                except RuntimeError:
                    print(
                        "Could not converge quasiparticle equation for"
                        + " sp_idx = {}, k_idx = {}, mo_idx = {}".format(
                            sp_idx, k_idx, mo_idx
                        )
                    )
                except ValueError:
                    print("Possible source of ValueError:")
                    print("Trying to access Sigma outside interpolation range.")
                    print(
                        " sp_idx = {}, k_idx = {}, mo_idx = {}".format(
                            sp_idx, k_idx, mo_idx
                        )
                    )
    
    return sc_qp_eigs
    
    
