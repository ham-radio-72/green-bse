#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    Functions for manipulating and contracting tensors.                      #
#                                                                             #


import h5py
import numpy as np
import scipy.linalg as LA
from irFT import IR_factory
# from irFT import tau2omegaFT, omega2tauFT
import gwtool as gw
import time


def readH5(filename, dataset):
    # read .h5 file.
    fi = h5py.File(filename, 'r')  
    tensor = fi[dataset][()].view(complex)
    return tensor

def readVQ(path="df_hf_int/VQ_0.h5"):
    # Only for one block of VQ at the moment. 
    VQ = readH5(path, "/0")

    return VQ

def readPtilde(filename = "p_iw_tilde_q0.h5"):
    # Read P tilde from h5 files.
    fi = h5py.File(filename, 'r')  
    iter = fi["/iter"][()]  # get the number of iteration.
    Ptilde = readH5(filename, "/iter{}/P_iw_tilde".format(iter))

    return Ptilde

def getPtilde_init(nQ,beta=1000,
                   tau_h5="1e5_120.h5", 
                   int_path="df_hf_int/",
                   input_h5="input.h5"):
    # Initialize P tilde from the initial G.
    P0 = gw.eval_P0_tilde_Q_init(nQ, beta=beta, 
                                 int_path=int_path, input_h5=input_h5, tau_h5=tau_h5)
    P0 = gw.symmetrize_P0(P0)
    # It is on the tau axis. This is evaluated with the Dyson equation.
    Ptilde_init = gw.eval_P_tilde(P0, tau_h5=tau_h5)

    # Bosonic Fourier transformation.
    # Ptau = tau2omegaFT(Ptilde)
    # print("Ptilde shape:", Ptilde_init.shape)
    # P = Ptilde.reshape(Ptilde.shape[0],1,Ptilde.shape[1],Ptilde.shape[2],1)

    return Ptilde_init

def getPtilde(iter,nao,nQ,tau_h5="1e5_120.h5", int_path="df_hf_int/", sim_h5="sim.h5"):
    # Compute P tilde from G and V.
    P0 = gw.eval_P0_tilde_Q(iter, nao, nQ, int_path=int_path, sim_h5=sim_h5)
    P0 = gw.symmetrize_P0(P0)
    Ptilde = gw.eval_P_tilde(P0, tau_h5=tau_h5)
    # print("Ptilde shape:", Ptilde.shape)

    return Ptilde

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

def getVPV(V,P):
    # Get VPV on iomega grid (or tilde W)
    print("*****    VPV    *****")
    start_time = time.time()
    niw = P.shape[0]
    nao = V.shape[2]
    nQ = V.shape[1]
    # print(P.shape)
    # print(V.shape)
    # PV = np.einsum("wqp,pkj->wqkj", P[:,0,:,:,0],V[0,:])
    # W = np.einsum("qil,wqkj->wilkj",V[0,:],PV)
    # W = W.transpose([0,1,4,3,2])  # wilkj -> wijkl
    PV = np.einsum("wqp,pkl->wqkl", P[:,0,:,:,0],V[0,:])
    W = np.einsum("qij,wqkl->wijkl",V[0,:],PV)
    W = W.transpose([0,1,3,2,4])
    # print(np.max(abs(W)))
    print("Evaluation of VPV finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    return W


def getW(V,P):
    # Get W on tau grid.
    print("*****    W    *****")
    VPV = getVPV(V,P)
    VV = np.einsum('qij,qkl->ijkl',V,V)
    for io in range(VPV.shape[0]):
        # VPV[io,:,:,:,:] += VV.transpose([0,2,1,3])
        VPV[io,:,:,:,:] += VV
    return VPV

def compute_mo(F, S, eigh_solver=LA.eigh, thr=1e-7):
    ns, nk, nao = F.shape[0:3]
    eiv_sk = np.zeros((ns, nk, nao))
    mo_coeff_sk = np.zeros((ns, nk, nao, nao), dtype=F.dtype)
    if S is None:
        S = np.array([[np.eye(nao)]*nk]*ns)
    for ss in range(ns):
        for k in range(nk):
            eiv, mo = eigh_solver(F[ss, k], S[ss, k], thr)
            # Re-order
            idx = np.argmax(abs(mo.real), axis=0)
            mo[:, mo[idx, np.arange(len(eiv))].real < 0] *= -1
            nbands = eiv.shape[0]
            eiv_sk[ss, k, :nbands] = eiv
            mo_coeff_sk[ss, k, :, :nbands] = mo
            # eiv_sk.append(eiv)
            # mo_coeff_sk.append(mo)
    # eig_sk = np.asarray(eiv_sk).reshape(ns, nk, nao)
    # mo_coeff_sk = np.asarray(mo_coeff_sk).reshape(ns, nk, nao, nao)

    return eiv_sk, mo_coeff_sk


def orth_mo(tensor_4p,Fock,S_ovlp):
    """
    Orthogonalizing matrix from AO to MO.
    """
    # Solve for MO eigenvectors.
    # Solve the generalized eigen problem: FC = SCE
    Fock = Fock.reshape(Fock.shape[:-1])
    S_ovlp = S_ovlp.reshape(S_ovlp.shape[:-1])
    fk_eigs, mo_vecs = compute_mo(Fock, S_ovlp)
    mo_vecs_adj = np.einsum('skba -> skab', mo_vecs.conj())
    s_c = np.einsum('skab, skbc -> skac', S_ovlp, mo_vecs)
    cdag_s = np.einsum('skab, skbc -> skac', mo_vecs_adj, S_ovlp)
    # Gt_ortho = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, G_tk_int, s_c, optimize=True)
    # resize nao^4 matrix into nao^2 using only the diagonal elements.
    dim = tensor_4p.shape[-1]
    ns  = tensor_4p.shape[0]
    nk  = tensor_4p.shape[1]
    
    temp  = np.zeros(tensor_4p.shape,dtype=np.complex128)  
    temp2 = np.zeros(tensor_4p.shape,dtype=np.complex128)  
    temp3 = np.zeros(tensor_4p.shape,dtype=np.complex128)  
    mat4P_MO = np.zeros(tensor_4p.shape,dtype=np.complex128)  
    for s in range(ns):
        for kp in range(nk):
            for i in range(0,dim):  
                for m in range(0,dim):  
                    temp[s,kp,i,:,:,:] += cdag_s[s,kp,i,m]*tensor_4p[s,kp,m,:,:,:]  
                for j in range(0,dim):  
                    for n in range(0,dim):  
                        temp2[s,kp,i,j,:,:] += s_c[s,kp,j,n]*temp[s,kp,i,n,:,:]  
                    for k in range(0,dim):  
                        for o in range(0,dim):  
                            temp3[s,kp,i,j,k,:] += cdag_s[s,kp,k,o]*temp2[s,kp,i,j,o,:]  
                        for l in range(0,dim):  
                            for p in range(0,dim):  
                                mat4P_MO[s,kp,i,j,k,l] += s_c[s,kp,l,p]*temp3[s,kp,i,j,k,p]  
    
    return mat4P_MO


def orth_mo_flatten(tensor_4p,Fock,S_ovlp):
    """
    Orthogonalizing matrix from AO to MO.
    """
    # Solve for MO eigenvectors.
    # Solve the generalized eigen problem: FC = SCE

    nao  = tensor_4p.shape[-1]
    npts = tensor_4p.shape[0]
    # Fock = Fock.reshape(Fock.shape[:-1])
    # S_ovlp = S_ovlp.reshape(S_ovlp.shape[:-1])
    fk_eigs, mo_vecs = compute_mo(Fock, S_ovlp)
    
    mo_vecs = mo_vecs.reshape(nao,nao)
    S_ovlp  = S_ovlp.reshape(nao,nao)

    C = S_ovlp @ mo_vecs
    C_p = mo_vecs.conj() @ S_ovlp

    C_4 = np.einsum("kc,dl->klcd",C,C_p).reshape(nao*nao,nao*nao)
    C_4_p = np.einsum("ia,bj->abij",C,C_p).reshape(nao*nao,nao*nao)
    
    # tensor_2D = tensor_4p.transpose([0,1,2,4,3]).reshape((npts,nao*nao,nao*nao))
    tensor_2D = tensor_4p.reshape((npts,nao*nao,nao*nao))

    tensor_2D_orth = np.einsum("im,tmn,nj->tij",C_4_p, tensor_2D, C_4) 
       
    return tensor_2D_orth

def orth_mo_2p(tensor_2p,Fock,S_ovlp):
    """
    Orthogonalizing matrix from AO to MO.
    """
    # Solve for MO eigenvectors.
    # Solve the generalized eigen problem: FC = SCE
    # print(tensor_2p.shape)
    Fock = Fock.reshape(Fock.shape[:-1])
    S_ovlp = S_ovlp.reshape(S_ovlp.shape[:-1])
    fk_eigs, mo_vecs = compute_mo(Fock, S_ovlp)
    mo_vecs_adj = np.einsum('skba -> skab', mo_vecs.conj())
    s_c = np.einsum('skab, skbc -> skac', S_ovlp, mo_vecs)
    cdag_s = np.einsum('skab, skbc -> skac', mo_vecs_adj, S_ovlp)
    # Gt_ortho = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, G_tk_int, s_c, optimize=True)
    # resize nao^4 matrix into nao^2 using only the diagonal elements.
    tensor_orth = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, tensor_2p, s_c, optimize=True)

    return tensor_orth


