#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Contraction function used in BSE.                                        # 
#                                                                             #

import h5py
import numpy as np
import scipy.linalg as LA
from ir import IR_factory,read_IR_matrices
import gwtool as gw
import time


def readPtilde(filename = "p_tau_tilde_q0.h5"):
    # Instead of calculating Ptilde from G and V, read it from a h5 file.
    fi = h5py.File(filename, 'r')  
    iter = fi["/iter"][()]  # get the number of iteration.
    Ptilde = readH5("p_tau_tilde_q0.h5", "/iter{}/P0_tau_tilde".format(iter))

    return Ptilde


def getPtilde(meta_h5 ="df_hf_int/meta.h5", \
              input_h5="input.h5", \
              sim_h5  ="scgw/sim.h5", \
              tau_h5  ="1e5_120.h5", \
              beta    =1000):
    # Compute P tilde from G and V.
    P0 = gw.eval_P0_tilde_Q(meta_h5, input_h5, sim_h5)
    P0 = gw.symmetrize_P0(P0)
    Ptilde = gw.eval_P_tilde(P0, beta, tau_h5)
    # Ptilde should be in tau axis.

    return Ptilde


def getPi(Gtau):
    # Get Pi on tau grid
    print("*****    Pi    *****")
    start_time = time.time()

    ntau = Gtau.shape[0]
    nao = Gtau.shape[3]
    Gtau_copy = Gtau.reshape(ntau,nao,nao)
    Pi = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for t in range(ntau):
        Pi[t,:,:,:,:] = -np.einsum("da,bc->abcd",Gtau_copy[t,:,:],Gtau_copy[ntau-t-1,:,:])

    print("Evaluation of Pi finished.")
    print("--- %s seconds ---" % (time.time() - start_time))

    return Pi


def staticUdiff(VQ, stateOption="total"):
    # - \sum_Q [V_{ij,Q}V_{kl,Q} - V_{ik,Q}V_{jl,Q}]
    print("*****    diff of U    *****")
    start_time = time.time()
    nao = VQ.shape[2]
    nQ = VQ.shape[1]
    Udiff = np.zeros((nao,nao,nao,nao),dtype=np.complex128)

    V = VQ.reshape(nQ,nao,nao)  # Q,i,j
    VV = np.einsum('qij,qkl->ijlk',V,V) # exchange 
    VV_2 = np.einsum('qij,qkl->ijkl',V,V) # coulomb
    if stateOption == "total":
        Udiff += (VV-VV_2)  
    if stateOption == "sing":
        Udiff += (2 * VV-VV_2)  
    if stateOption == "trip":
        Udiff += (-VV_2)  
        
    print("Evaluation of U difference finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return Udiff


def getXi(stateOption="total" ,readP = False):
    """
    Xi = U - W
    """
    V = readVQ()
    nao = V.shape[2]
    if readP:
        P = readPtilde()
    else:
        P = getPtilde(iter, V.shape[2], V.shape[1])

    ntau = P.shape[0]
    Xi = -getVPV(V,P)
    if stateOption=="total":
        Udiff = staticUdiff(V, "total")
    if stateOption=="sing":
        Udiff = staticUdiff(V, "sing")
    if stateOption=="trip":
        Udiff = staticUdiff(V, "trip")
    
    for t in range(ntau):
        Xi[t,:,:,:,:] += Udiff
    
    return Xi


def updateChi(Chi, Xi, Pi):
    """
    Update the Chi matrix via BSE:
    Chi(new) = Pi + Pi Xi Chi(old).
    """
    ntau = Chi.shape[0]
    for t in range(ntau):
        Chi[t,:,:,:,:] = Pi[t,:,:,:,:] + \
                         np.einsum("ijqp,pqkl->ijkl",Pi[t,:,:,:,:],
                                   np.einsum("pqnm,mnkl->pqkl",Xi[t,:,:,:,:],Chi[t,:,:,:,:]))

    return Chi


def getVPV(V,P):
    # VPV, or screened Coulomb interaction.
    # Get VPV on tau grid.
    print("*****    VPV    *****")
    start_time = time.time()
    ntau = P.shape[0]
    nao = V.shape[2]
    nQ = V.shape[1]
    VPV = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for it in range(ntau):
        Pomega = P[it,0,:,:,0].reshape(nQ,nQ)
        V1 = V.reshape(nQ,nao,nao) # i,k,Q 
        X1 = np.einsum('qij,qp->ijp',V1,Pomega)
        X2 = np.einsum('ijp,pkl->ijkl',X1,V1)
        VPV[it,:,:,:,:] += X2
    
    print("Evaluation of VPV finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    return VPV


def getUChi_2p(Chi,beta,ir_h5):
    # For molecular microscopic and macroscopic epsilon.
    print("*****    UChi    *****")
    start_time = time.time()
    ntau = Chi.shape[0]
    nao = Chi.shape[1]
    
    print("Contracting Chi to two points...")
    Chi2p = contractChi2p(Chi)
    
    VQ = readVQ()
    nQ = VQ.shape[1]
    V = VQ.reshape(nQ,nao,nao)  # Q,i,j
    U = np.einsum('qab,qcd->abcd',V,V)
    
    UChi_tau = np.zeros(Chi2p.shape,dtype=np.complex128)
    for t in range(ntau):
        UChi_tau[t,:,:] += (np.einsum('ijkl,jl->ik',U,Chi2p[t,:,:]))
    
    print("F-T UChi to imag frequency...")
    UChi_iomega = tau2omegaFT(UChi_tau,beta,ir_h5)
    print("Evaluation of UChi finished.")    
    print("--- %s seconds ---" % (time.time() - start_time))

    return UChi_iomega


def getMicroEps(UChi):
    """
    Input UChi (un-orthogonalized) on the iomega axis.
    """
    nao = UChi.shape[-1]
    nfreq = UChi.shape[0]

    microEps = np.zeros(UChi.shape,dtype=np.complex128)

    for io in range(nfreq):
        microEps[io,:,:] += UChi[io,:,:] + np.eye(nao)
    
    return microEps


def getMacroEps(microEps):
    """
    Input microEps (un-orthogonalized) on the iomega axis.
    """
    
    print("Inverting microEps to macroEps...")
    nfreq = microEps.shape[0]
    nao = microEps.shape[-1]
    macroEps = np.zeros(microEps.shape,dtype=np.complex128)
    for io in range(nfreq):
        macroEps[io,:,:] += LA.inv(microEps[io,:,:])

    return macroEps


def tau2omegaFT(tau, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img time to img freq...")
    fourier = IR_factory(beta, tau_h5)
    omega = fourier.tauf_to_wb(tau)
    
    return omega


def tau2omegaFTforG(tau, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    # For G or any Fermionic quantity.
    print("Performing Fourier transformation from img time to img freq...")
    fourier = IR_factory(beta, tau_h5)
    omega = fourier.tau_to_w(tau)
    
    return omega

def omega2tauFT(omega, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img freq to img time...")
    fourier = IR_factory(beta, tau_h5)
    tau = fourier.wb_to_tauf(omega)
    
    return tau

def omega2tauFTforG(omega, beta = 1000, tau_h5 = "/home/wenm/irgrids/" + "1e5_120.h5"):
    print("Performing Fourier transformation from img freq to img time...")
    fourier = IR_factory(beta, tau_h5)
    tau = fourier.w_to_tau(omega)
    
    return tau