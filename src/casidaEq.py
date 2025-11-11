#!/usr/bin/env python3
#                                                                             #
#    Copyright (c) 2025 Ming Wen <wenm@umich.edu>, University of Michigan.    #
#                                                                             #
#    The Casida equation functions for BSE.                                   #
#    Parallelized frequency loops for improved performance.                   #
#                                                                             #


import numpy as np
import contract as ct
import h5py
import scipy.linalg as LA
from joblib import Parallel, delayed 


def matEleA(diffEps, VQ, W):
    """Calculate A matrix element for BSE.
    A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    """
    U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :])
    A = diffEps + U - W.transpose([0, 2, 1, 3])
    
    return A


def matEleB(VQ, W):
    """Calculate B matrix element for BSE.
    B(ia,jb) = U(ia,bj) - W(ib,aj)
    """
    U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :])
    nao = W.shape[-1]
    # B = U.transpose([0,1,3,2]) - W.transpose([0,3,1,2])
    B = np.zeros(W.shape, dtype=np.complex128)
    for a in range(nao):
        for j in range(nao):
            for b in range(nao):
                B[:, a, j, b] = U[:, a, b, j] - W[:, b, a, j]

    return B


def matEleADyn(diffEps, VQ, W):
    """Calculate dynamic A matrix element for BSE.
    A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    """
    U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :])
    A = np.zeros(W.shape, dtype=np.complex128)
    nump = W.shape[0]
    for p in range(nump):
        W_temp = W[p, :]
        A[p, :] += diffEps + U - W_temp.transpose([0, 2, 1, 3])
    
    return A


def matEleXiDyn(VQ, W, type="normal"):
    """Calculate dynamic Xi matrix element for BSE.
    A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    """
    A = np.zeros(W.shape, dtype=np.complex128)
    nump = W.shape[0]
    
    if (type != "singlet") and (type != "triplet"):
        U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :])
        for p in range(nump):
            W_temp = W[p, :]
            A[p, :] += U - W_temp.transpose([0, 2, 1, 3])
            # A[p,:] += U - W_temp
            # A[p,:] += U.transpose([0,1,3,2]) - W_temp.transpose([0,3,2,1])
    elif type == "singlet":
        U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :])
        for p in range(nump):
            W_temp = W[p, :]
            A[p, :] += 2 * U - W_temp.transpose([0, 2, 1, 3])
            # A[p,:] += 2 * U - W_temp
            # A[p,:] += 2 * U.transpose([0,1,3,2]) - W_temp.transpose([0,3,2,1])
    elif type == "triplet":
        for p in range(nump):
            W_temp = W[p, :]
            A[p, :] += -W_temp.transpose([0, 2, 1, 3])
            # A[p,:] += - W_temp
            # A[p,:] += - W_temp.transpose([0,3,2,1])
    
    return A


def matEleBDyn(VQ, W, type="normal"):
    """Calculate dynamic B matrix element for BSE.
    B(ia,jb) = Eps(ia,bj) - W(ib,aj)
    """
    # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    B = np.zeros(W.shape, dtype=np.complex128)
    nump = W.shape[0]
    nao = W.shape[-1]
    # for p in range(nump):
    #     W_temp = W[p,:]
    #     B[p,:] += U.transpose([0,1,3,2]) - W_temp.transpose([0,3,1,2])
    
    if (type != "singlet") and (type != "triplet"):
        U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :], optimize='optimal')
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p, :, a, j, b] = U[:, a, b, j] - W[p, :, b, a, j]
                        # B[p,:,a,j,b] = U[:,a,b,j] - W[p,:,a,b,j]
                        # B[p,:,a,j,b] = U[:,a,j,b] - W[p,:,j,b,a]
    elif type == "singlet":
        U = np.einsum('qia,qjb->iajb', VQ[0, :], VQ[0, :], optimize='optimal')
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p, :, a, j, b] = 2 * U[:, a, b, j] - W[p, :, b, a, j]
                        # B[p,:,a,j,b] = 2 * U[:,a,b,j] - W[p,:,a,b,j]
                        # B[p,:,a,j,b] = 2 * U[:,a,j,b] - W[p,:,j,b,a]
    elif type == "triplet":
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p, :, a, j, b] = -W[p, :, b, a, j]
                        # B[p,:,a,j,b] = - W[p,:,a,b,j]
                        # B[p,:,a,j,b] = - W[p,:,j,b,a]

    return B


def matEleXiStat(VQ, W, nelec, type="normal"):
    """Calculate static Xi matrix element for BSE.
    A(ia,jb) = Eps(ia,jb) + U(ia,bj) - W(ij,ab)
    """
    occ = nelec // 2
    virt = VQ.shape[-1] - occ
    # i (occ), a (virt), j (occ), b (virt)
    Xi = np.zeros((occ, virt, occ, virt), dtype=np.complex128)
    
    if (type != "singlet") and (type != "triplet"):
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, :occ, occ:], optimize='optimal')
        Xi += U - W.transpose([0, 2, 1, 3])
    elif type == "singlet":
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qjb->iajb', VQ[0, :, :occ, occ:], VQ[0, :, :occ, occ:], optimize='optimal')
        Xi += 2 * U - W.transpose([0, 2, 1, 3])
    elif type == "triplet":
        Xi += -W.transpose([0, 2, 1, 3])
    
    # Xi = mo2ovStat(Xi)
    return Xi


def matEleBStat(VQ, W, nelec, type="normal"):
    """Calculate static B matrix element for BSE.
    B(ia,jb) = U(ia,bj) - W(ib,aj)
    """
    nao = VQ.shape[-1]
    occ = nelec // 2
    virt = nao - occ
    B = np.zeros((occ, virt, occ, virt), dtype=np.complex128)
    
    if (type != "singlet") and (type != "triplet"):
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize="optimal")
        # U shape is (occ,virt,virt,occ)
        # B shape is (occ,virt,occ,virt)
        U = np.einsum('qia,qbj->iabj', VQ[0, :, :occ, occ:], VQ[0, :, occ:, :occ], optimize='optimal')
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:, a, j, b] = U[:, a, b, j] - W[:, b, a, j]
    elif type == "singlet":
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qbj->iabj', VQ[0, :, :occ, occ:], VQ[0, :, occ:, :occ], optimize='optimal')
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:, a, j, b] = 2 * U[:, a, b, j] - W[:, b, a, j]
    elif type == "triplet":
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:, a, j, b] = -W[:, b, a, j]

    # B = mo2ovStat(B)
    return B


def getW(P, V):
    """Calculate W matrix using VPV form.
    W is in the exchange form.
    """
    print("*****    VPV    *****")
    print(P.shape)
    print(V.shape)
    
    iwpts = P.shape[0]
    nao = V.shape[-1]

    # U = np.einsum('qil,qkj->ijkl',V[0,:],V[0,:],optimize='optimal')
    U = np.einsum('qij,qkl->ijkl',V[0,:],V[0,:],optimize='optimal')
    W = np.zeros((iwpts,nao,nao,nao,nao),dtype=U.dtype)
    print(W.shape)
    
    print("*****    U    *****")
    for w in range(iwpts):
        PV = np.einsum("qp,pkl->qkl", P[w,0,:,:,0], V[0,:], optimize='optimal')
        VPV = np.einsum("qij,qkl->ijkl", V[0,:], PV, optimize='optimal')
        W[w,:] = VPV + U
    
    return W


def solveMO(F, S, eigh_solver=LA.eigh, thr=1e-7):
    # print("*****    Solving Fock    *****")
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

    return eiv_sk, mo_coeff_sk


def flattenOrth(mat4p,Fock,S_ovlp):
    if len(mat4p.shape) != 4:
        raise Exception("Matrix is not in four-point.")
    
    nao = mat4p.shape[-1]
    _,mo_vecs = solveMO(Fock,S_ovlp)
    
    mo_vecs = mo_vecs.reshape(nao,nao)
    S_ovlp  = S_ovlp.reshape(nao,nao)
    C = S_ovlp @ mo_vecs
    C_p = mo_vecs.conj() @ S_ovlp

    C_4 = np.einsum("kc,dl->klcd",C,C_p).reshape(nao*nao,nao*nao)
    C_4_p = np.einsum("ia,bj->abij",C,C_p).reshape(nao*nao,nao*nao)

    # chi_tau_2D_orth = C_4.conj() @ chi_tau_2D[0,:] @ C_4
    mat4p_2D = mat4p.reshape(nao*nao,nao*nao)
    mat4p_2D_orth = np.einsum("im,mn,nj->ij", C_4_p, mat4p_2D, C_4)

    return mat4p_2D_orth


def VQ_ao2mo(VQ, mo_vecs):
    """
    Transform the V_{Qij} tensor from AO basis to MO basis.
    """
    # dimmension of VQ is (1, nQ, nao, nao)
    VQ_mo  = np.zeros(VQ.shape,dtype=np.complex128)  
    for ik in range(VQ.shape[0]):
        for iQ in range(VQ.shape[1]):
            # Step 1: temp = M_ao[ik, iQ] @ C  -> shape (nao, nmo)
            temp = VQ[ik, iQ] @ mo_vecs
            # Step 2: M_mo[ik, iQ] = C.T @ temp -> shape (nmo, nmo)
            VQ_mo[ik, iQ] = mo_vecs.conj().T @ temp

    return VQ_mo


def ao2ov(mat4P,Fock,S_ovlp,nelec):
    
    nao = mat4P.shape[-1]
    _,mo_vecs = solveMO(Fock,S_ovlp)
    mo_vecs = mo_vecs.reshape(nao,nao)
    occ  = nelec//2
    virt = nao - occ
    # separate the mo_vecs matrix into two blocks.
    # moVex_occ  = mo_vecs[:occ,:]       # size = nao * occ
    # moVex_virt = mo_vecs[occ:nao,:]    # size = nao * virt   
    moVex_occ  = mo_vecs[:,:occ]       # size = nao * occ
    moVex_virt = mo_vecs[:,occ:]    # size = nao * virt
    
    temp  = np.zeros((occ,nao,nao,nao),dtype=np.complex128)  
    temp2 = np.zeros((occ,virt,nao,nao),dtype=np.complex128)  
    temp3 = np.zeros((occ,virt,occ,nao),dtype=np.complex128)  
    mat4P_MO = np.zeros((occ,virt,occ,virt),dtype=np.complex128)  
    
    # i & j are occpied orbs
    # a & b are virtual orbs
    # iajb -> mn, m = n = virt * occ
    
    for i in range(0,occ):  
        for m in range(0,nao):  
            temp[i,:,:,:] += moVex_occ.T[i,m]*mat4P[m,:,:,:]  
        for a in range(0,virt):  
            for n in range(0,nao):  
                temp2[i,a,:,:] += moVex_virt[n,a]*temp[i,n,:,:]  
            for j in range(0,occ):  
                for o in range(0,nao):  
                    temp3[i,a,j,:] += moVex_occ.T[j,o]*temp2[i,a,o,:]  
                for b in range(0,virt):  
                    for p in range(0,nao):  
                        mat4P_MO[i,a,j,b] += moVex_virt[p,b]*temp3[i,a,j,p]  
    
    return mat4P_MO


def ao2ovDyn(mat4PDyn,Fock,S_ovlp,nelec):
    
    nao = mat4PDyn.shape[-1]
    _,mo_vecs = solveMO(Fock,S_ovlp)
    mo_vecs = mo_vecs.reshape(nao,nao)
    occ  = nelec//2
    virt = nao - occ
    nump = mat4PDyn.shape[0]
    # separate the mo_vecs matrix into two blocks.
    moVex_occ  = mo_vecs[:,:occ]       # size = nao * occ
    moVex_virt = mo_vecs[:,occ:]    # size = nao * virt
    
    temp  = np.zeros((occ,nao,nao,nao),dtype=np.complex128)  
    temp2 = np.zeros((occ,virt,nao,nao),dtype=np.complex128)  
    temp3 = np.zeros((occ,virt,occ,nao),dtype=np.complex128)  
    
    mat4P_MO = np.zeros((nump,occ,virt,occ,virt),dtype=np.complex128)  
    
    # i & j are occpied orbs
    # a & b are virtual orbs
    # iajb -> mn, m = n = virt * occ
    for pt in range(nump):
        temp  = np.zeros((occ,nao,nao,nao),dtype=np.complex128)  
        temp2 = np.zeros((occ,virt,nao,nao),dtype=np.complex128)  
        temp3 = np.zeros((occ,virt,occ,nao),dtype=np.complex128)  
        for i in range(0,occ):  
            for m in range(0,nao):  
                temp[i,:,:,:] += moVex_occ.T[i,m]*mat4PDyn[pt,m,:,:,:]  
            for a in range(0,virt):  
                for n in range(0,nao):  
                    temp2[i,a,:,:] += moVex_virt[n,a]*temp[i,n,:,:]  
                for j in range(0,occ):  
                    for o in range(0,nao):  
                        temp3[i,a,j,:] += moVex_occ.T[j,o]*temp2[i,a,o,:]  
                    for b in range(0,virt):  
                        for p in range(0,nao):  
                            mat4P_MO[pt,i,a,j,b] += moVex_virt[p,b]*temp3[i,a,j,p]  
        
    return mat4P_MO


def mo2ovDyn(mat4PDyn_mo,nelec):
    # i & j are occpied orbs
    # a & b are virtual orbs
    # iajb -> mn, m = n = virt * occ
    occ  = nelec//2
    mat4PDyn_ov = mat4PDyn_mo[:,0:occ,occ:,0:occ,occ:]
    print(mat4PDyn_ov.shape)

    return mat4PDyn_ov


def mo2ovStat(mat4PStat_mo,nelec):
    occ  = nelec//2
    mat4PStat_ov = mat4PStat_mo[:occ,occ:,:occ,occ:]

    return mat4PStat_ov


def diffEpsMat(eigVals):
    nao = len(eigVals)
    diffEps = np.zeros((nao,nao,nao,nao),dtype=np.complex128)
    # ia,jb
    # i = j and a = b
    for i in range(nao):
        for a in range(nao):
            diffEps[i,a,i,a] += eigVals[a] - eigVals[i]
        
    return diffEps


def concatAB(A,B):
    """
    Concatenate two matrices A and B into a block matrix.
    For static version.
    """
    block_dim = A.shape[-1]
    temp = np.zeros((2*block_dim,2*block_dim),dtype=np.complex128)
    temp[0:block_dim,0:block_dim] += A
    temp[block_dim:,0:block_dim] += -B.conj()
    temp[0:block_dim,block_dim:] += B
    temp[block_dim:,block_dim:] += -A.conj()

    return temp


def concatABDyn(A,B):
    """
    Concatenate two matrices A and B into a block matrix.
    For dynamic version.
    """
    block_dim = A.shape[-1]
    nump = A.shape[0]
    temp = np.zeros((nump,2*block_dim,2*block_dim),dtype=np.complex128)
    temp[:,0:block_dim,0:block_dim] += A
    temp[:,block_dim:,0:block_dim] += -B.conj()
    temp[:,0:block_dim,block_dim:] += B
    temp[:,block_dim:,block_dim:] += -A.conj()

    return temp


def getDiffEps(valsMO,nelec):
    nao   = valsMO.shape[-1]
    occ   = nelec // 2
    virt  = nao - occ
    diffEps = diffEpsMat(valsMO[0,0,:])
    diffEps_ov = np.zeros([occ*virt,occ*virt],dtype=np.complex128)
    for i in range(occ):
        for a in range(occ,nao):
            # iajb -> ia,jb
            ia = i * virt + a - occ
            diffEps_ov[ia, ia] += diffEps[i, a, i, a] 

    return diffEps_ov


def HinfDiagApprox(effVex,VQ,valsMO,nelec,ex_type="singlet"):
    # Solve the Hamiltonian (iomega -> infinity) 
    # Must be in MO basis already.
    # return the diagonalized H_inf matrix.
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:]),nelec).reshape(occ*virt,occ*virt)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    H2p_inf = np.zeros((2*occ*virt),dtype=np.complex128)

    W_inf   = U1
    A   = matEleXiStat(VQ,W_inf,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    W_inf   = U2
    B   = matEleBStat(VQ,W_inf,nelec,ex_type)
    B   = B.reshape(occ*virt,occ*virt)
    H   = concatAB(A,B)
    # Not necessarily Hermitian.
    effVex_inv = LA.inv(effVex)
    
    H2p_inf += np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)

    return H2p_inf


def getHinf(VQ,valsMO,nelec,ex_type="singlet"):
    # Solve the Hamiltonian (iomega -> infinity) 
    # Must be in MO basis already.
    # return the diagonalized H_inf matrix.
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:]),nelec).reshape(occ*virt,occ*virt)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    H2p_inf = np.zeros((2*occ*virt),dtype=np.complex128)

    W_inf = U1
    A = matEleXiStat(VQ,W_inf,nelec,ex_type)
    A = diffEps_ov + A.reshape(occ*virt,occ*virt)
    W_inf = U2
    B = matEleBStat(VQ,W_inf,nelec,ex_type)
    B = B.reshape(occ*virt,occ*virt)
    H2p_inf = concatAB(A,B)
    # Not necessarily Hermitian.

    return H2p_inf


def initG2p_inv(H2p_inf,ir_file,beta=1000):
    # Initial diagonlaized G2p from H2p at iomega -> inf.
    wgrid = h5py.File(ir_file,"r")["/bose/wsample"][()]
    wgrid = 2 * wgrid * np.pi / beta
    niw = len(wgrid)

    G2p_inv_init = np.zeros((niw,H2p_inf.shape[0]),dtype=np.complex128)
    for iw in range(niw):
        G2p_inv_init[iw,:] = (1j * wgrid[iw] - H2p_inf)

    return G2p_inv_init
    
    
def _process_w2p_frequency(iw, Pi, VQ, effVex_inv, effVex, occ, virt):
    """Helper function to process W2p for a single frequency point."""
    # PV  = np.einsum("qp,pkl->qkl", Pi[iw,:], VQ[0,:,occ:,occ:], optimize='optimal') 
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    A   = -VPV.transpose([0,2,1,3])
    A   = A.reshape(occ*virt,occ*virt)
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
    B   = -np.einsum("ibaj->iajb",VPV) 
    B   = B.reshape(occ*virt,occ*virt)
    W2p = concatAB(A,B)
    return np.einsum('ij,jk,ki->i', effVex_inv, W2p, effVex)


def getW2p_tilde(Pi,VQ,effVex,nelec,n_jobs=-1):
    # Must be in MO basis already.
    # Only keep the diagonal elements.
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    print(VQ.shape)
    niw  = Pi.shape[0] 
    niw_half = niw // 2 + 1
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    effVex_inv = LA.inv(effVex)
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_w2p_frequency)(
            iw, Pi, VQ, effVex_inv, effVex, occ, virt
        ) for iw in range(niw_half)
    )
    
    # Collect results (always symmetric)
    W2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        W2p_Dyn[iw] = result
    for iw, result in enumerate(results):
        W2p_Dyn[niw - 1 - iw] = result
    
    return W2p_Dyn


def getSigma2p(G2p_init,W2p_Dyn):
    # remember to invert G2p_inv_init first.
    # All quantities have to be diagonal.
    # calculate Sigma2p in imag frequency domain.
    Sigma2p = np.zeros(W2p_Dyn.shape,dtype=np.complex128)
    niw = G2p_init.shape[0]
    for w in range(niw):
        Sigma2p[w,:] += 1/(1 - G2p_init[w,:] * W2p_Dyn[w,:])
        Sigma2p[w,:] *= W2p_Dyn[w,:]

    return Sigma2p


def _process_frequency_point(iw, G2p_iw_inv, Pi, effVex_inv, effVex, VQ, occ, virt):
    """Helper function to process a single frequency point."""
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    A   = -VPV.transpose([0,2,1,3])
    A   = A.reshape(occ*virt,occ*virt)
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
    B   = -np.einsum("ibaj->iajb",VPV) 
    B   = B.reshape(occ*virt,occ*virt)
    W2p = concatAB(A,B)
    W2p = np.einsum('ij,jk,kl->il', effVex_inv, W2p, effVex, optimize='optimal') 
    # calculate Sigma2p in imag frequency domain.
    G2p = 1.0/G2p_iw_inv[iw,:]
    temp = G2p * W2p
    temp = np.eye(2*occ*virt,dtype=np.complex128) - temp
    # (I-GH)^-1 
    temp = LA.inv(temp)
    # H(I-GH)^-1
    Sigma2p = np.einsum('ij,jk->ik',  W2p, temp, optimize='optimal') 
    return np.diag(LA.inv(np.diag(G2p_iw_inv[iw]) - Sigma2p))


def updateG2p_alt(G2p_iw_inv,Pi,effVex,VQ,nelec,n_jobs=-1):
    # Must be in MO basis already.
    # Only keep the diagonal elements.
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    niw  = Pi.shape[0]  # always odd
    niw_half = niw // 2 + 1
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    effVex_inv = LA.inv(effVex)
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points (half)
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_frequency_point)(
            iw, G2p_iw_inv, Pi, effVex_inv, effVex, VQ, occ, virt
        # ) for iw in range(niw)
        ) for iw in range(niw_half)
    )
    
    # Collect results (always symmetric)
    G2p_updated_iw = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        G2p_updated_iw[iw] = result
    for iw, result in enumerate(results):
        G2p_updated_iw[niw - 1 - iw] = result
    
    return G2p_updated_iw


def updateG2p(G2p_inv_init,W2p_tilde,ir_file,beta=1000):
    # both G2p_inv_init and Sigma2p_tau
    G2p_updated_iw = np.zeros(G2p_inv_init.shape,dtype=np.complex128)
    G2p_init = 1.0/G2p_inv_init
    Sigma2p_iw = getSigma2p(G2p_init,W2p_tilde)
    for iw in range(G2p_inv_init.shape[0]):
        G2p_updated_iw[iw,:] = 1/(G2p_inv_init[iw,:] - Sigma2p_iw[iw,:])

    return G2p_updated_iw


def _process_hdyn_frequency(iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1, U2):
    """Helper function to process HDyn for a single frequency point."""
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    W   = VPV + U1
    A   = matEleXiStat(VQ,W,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
    # ijkl: Nocc * Nvirt * Nvirt * Nocc
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
    # ijkl: Nocc * Nvirt * Nvirt * Nocc
    W   = VPV + U2
    B   = matEleBStat(VQ,W,nelec,ex_type)
    B   = B.reshape(occ*virt,occ*virt)
    H   = concatAB(A,B)
    return np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)


def HDynDiagApprox(Pi,VQ,valsMO,nelec,ex_type="singlet",n_jobs=-1):
    # Must be in MO basis already.
    # reorder from lowet to highest in case of energy reshuffle.
    occ  = nelec // 2
    virt = VQ.shape[-1] - occ
    diffEps_ov = mo2ovStat(diffEpsMat(valsMO[0,0,:]),nelec).reshape(occ*virt,occ*virt)
    niw  = Pi.shape[0] 
    niw_half = niw // 2 + 1
    effVals, effVex = solveHstatic(Pi[niw//2,0,:,:,0],VQ,diffEps_ov,nelec,ex_type,tda=0)
    effVex_inv = LA.inv(effVex)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_hdyn_frequency)(
            iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1, U2
        ) for iw in range(niw_half)
    )
    
    # Collect results (always symmetric)
    H2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_Dyn[iw] = result
    for iw, result in enumerate(results):
        H2p_Dyn[niw - 1 - iw] = result
        
    # return effVals, H2p_Dyn, valsMO
    return effVals, H2p_Dyn


def _process_hdyn_tda_frequency(iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1):
    """Helper function to process HDyn TDA for a single frequency point."""
    # PV  = np.einsum("qp,pkl->qkl", Pi[iw,:], VQ[0,:,occ:,occ:], optimize='optimal') 
    PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    W   = VPV + U1
    A   = matEleXiStat(VQ,W,nelec,ex_type)
    A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
    B   = np.zeros((occ*virt,occ*virt),dtype=np.complex128) 
    H   = concatAB(A,B)
    return np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)


def HDynDiagApprox_TDA(Pi,VQ,valsMO,nelec,ex_type="singlet",n_jobs=-1):
    # Must be in MO basis already.
    diffEps_ov = getDiffEps(valsMO,nelec)
    niw  = Pi.shape[0] 
    niw_half = niw // 2 + 1
    effVals, effVex = solveHstatic(Pi[niw//2,0,:,:,0],VQ,diffEps_ov,nelec,ex_type,tda=1)
    effVex_inv = LA.inv(effVex)
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    # Diagonal approximation to H2p_Dyn at each frequency point.
    # Parallelize over frequency points
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(_process_hdyn_tda_frequency)(
            iw, Pi, VQ, effVex_inv, effVex, occ, virt, diffEps_ov, nelec, ex_type, U1
        ) for iw in range(niw_half)
    )

    # Collect results (always symmetric)
    H2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw, result in enumerate(results):
        H2p_Dyn[iw] = result
    for iw, result in enumerate(results):
        H2p_Dyn[niw - 1 - iw] = result

    return effVals, H2p_Dyn


def solveHstatic(Pi_stat,VQ,diffEps_ov,nelec,ex_type="singlet",tda=0):
    # Must be in MO basis already.
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    # print("*****    Solving static effective Hamiltonian    *****")
    # Qkl: NQ * Nvirt * Nvirt
    PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    W_stat  = VPV + U
    # iajb: Nocc * Nvirt * Nocc * Nvirt
    A_stat = matEleXiStat(VQ,W_stat,nelec,ex_type)
    
    if not tda:
        # Qkl: NQ * Nvirt * Nocc
        PV  = np.einsum("qp,pkl->qkl", Pi_stat, VQ[0,:,occ:,:occ], optimize='optimal') 
        # ijkl: Nocc * Nvirt * Nvirt * Nocc
        VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
        # ijkl: Nocc * Nvirt * Nvirt * Nocc 
        U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
        W_stat  = VPV + U
        B_stat = matEleBStat(VQ,W_stat,nelec,ex_type)
    else:
        print("TDA approximation is enabled. Using TDA effective Hamiltonian.")
        B_stat = np.zeros((occ,virt,occ,virt),dtype=np.complex128)

    A_stat  = diffEps_ov + A_stat.reshape(occ*virt,occ*virt)
    B_stat  = B_stat.reshape(occ*virt,occ*virt)
    
    H_stat  = concatAB(A_stat,B_stat)
    effVals,effVex = LA.eig(H_stat)
    # renormalize eigenvectors
    o_idx_li = []
    v_idx_li = []
    for i in range(len(effVals)):
        effVex[:,i] /= LA.norm(effVex[:,i])
        ov_idx = np.argmax(effVex[:,i] ** 2)
        if ov_idx >= (occ * virt):
            ov_idx -= (occ * virt)
        o_idx = ov_idx // virt
        v_idx = occ + ov_idx % virt
        o_idx_li.append(o_idx)
        v_idx_li.append(v_idx)
    o_idx_li = np.array(o_idx_li)
    v_idx_li = np.array(v_idx_li)

    idx = np.argsort(np.abs(effVals))
    effVals = effVals[idx]
    effVex  = effVex[:,idx]
    o_idx_li = o_idx_li[idx]
    v_idx_li = v_idx_li[idx]

    return effVals, effVex, o_idx_li, v_idx_li