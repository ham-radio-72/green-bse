import numpy as np
import contract as ct
import h5py
import scipy.linalg as LA


def matEleA(diffEps,VQ,W):
    # A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    A = diffEps + U - W.transpose([0,2,1,3])
    
    return A


def matEleB(VQ,W):
    # B(ia,jb) = U(ia,bj) - W(ib,aj)
    U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    nao = W.shape[-1]
    # B = U.transpose([0,1,3,2]) - W.transpose([0,3,1,2])
    B = np.zeros(W.shape,dtype=np.complex128)
    for a in range(nao):
        for j in range(nao):
            for b in range(nao):
                B[:,a,j,b] = U[:,a,b,j] - W[:,b,a,j]

    return B


def matEleADyn(diffEps,VQ,W):
    # A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    A = np.zeros(W.shape,dtype=np.complex128)
    nump = W.shape[0]
    for p in range(nump):
        W_temp = W[p,:]
        A[p,:] += diffEps + U - W_temp.transpose([0,2,1,3])
    
    return A


def matEleXiDyn(VQ,W,type="normal"):
    # A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    A = np.zeros(W.shape,dtype=np.complex128)
    nump = W.shape[0]
    if (type != "singlet") and (type != "triplet"):
        U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
        for p in range(nump):
            W_temp = W[p,:]
            A[p,:] += U - W_temp.transpose([0,2,1,3])
    elif type == "singlet":
        U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
        for p in range(nump):
            W_temp = W[p,:]
            A[p,:] += 2 * U - W_temp.transpose([0,2,1,3])
    elif type == "triplet":
        for p in range(nump):
            W_temp = W[p,:]
            A[p,:] += - W_temp.transpose([0,2,1,3])
    
    return A


def matEleXiStat(VQ,W,nelec,type="normal"):
    # A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    occ = nelec//2
    virt = VQ.shape[-1] - occ
    Xi = np.zeros((occ,virt,occ,virt),dtype=np.complex128)
    
    if (type != "singlet") and (type != "triplet"):
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qjb->iajb',VQ[0,:,:occ,occ:],VQ[0,:,:occ,occ:],optimize='optimal')
        Xi += U - W.transpose([0,2,1,3])
    elif type == "singlet":
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qjb->iajb',VQ[0,:,:occ,occ:],VQ[0,:,:occ,occ:],optimize='optimal')
        Xi += 2 * U - W.transpose([0,2,1,3])
    elif type == "triplet":
        Xi += - W.transpose([0,2,1,3])
    
    # Xi = mo2ovStat(Xi)
    return Xi


def matEleBDyn(VQ,W,type="normal"):
    # B(ia,jb) = Eps(ia,bj) - W(ib,aj)
    # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    B = np.zeros(W.shape,dtype=np.complex128)
    nump = W.shape[0]
    nao  = W.shape[-1]
    # for p in range(nump):
    #     W_temp = W[p,:]
    #     B[p,:] += U.transpose([0,1,3,2]) - W_temp.transpose([0,3,1,2])
    if (type != "singlet") and (type != "triplet"):
        U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p,:,a,j,b] = U[:,a,b,j] - W[p,:,b,a,j]
    elif type == "singlet":
        U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p,:,a,j,b] = 2 * U[:,a,b,j] - W[p,:,b,a,j]
    elif type == "triplet":
        for p in range(nump):
            for a in range(nao):
                for j in range(nao):
                    for b in range(nao):
                        B[p,:,a,j,b] = - W[p,:,b,a,j]

    return B


def matEleBStat(VQ,W,nelec,type="normal"):
    # B(ia,jb) = Eps(ia,bj) - W(ib,aj)
    nao  = VQ.shape[-1]
    occ  = nelec//2
    virt = nao - occ
    B = np.zeros((occ,virt,occ,virt),dtype=np.complex128)
    if (type != "singlet") and (type != "triplet"):
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize="optimal")
        U = np.einsum('qia,qjb->iajb',VQ[0,:,:occ,occ:],VQ[0,:,occ:,:occ],optimize='optimal')
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:,a,j,b] = U[:,a,b,j] - W[:,b,a,j]
    elif type == "singlet":
        # U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:],optimize='optimal')
        U = np.einsum('qia,qjb->iajb',VQ[0,:,:occ,occ:],VQ[0,:,occ:,:occ],optimize='optimal')
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:,a,j,b] = 2 * U[:,a,b,j] - W[:,b,a,j]
    elif type == "triplet":
        for a in range(virt):
            for j in range(occ):
                for b in range(virt):
                    B[:,a,j,b] = - W[:,b,a,j]

    # B = mo2ovStat(B)
    return B


def getW(P,V):
    print("*****    VPV    *****")
    print(P.shape)
    print(V.shape)
    
    iwpts = P.shape[0]
    nao   = V.shape[-1]
    # examine efficiency of einsum.
    # PV = np.einsum("wqp,pkl->wqkl", P[:,0,:,:,0],V[0,:],optimize='optimal')
    # W = np.einsum("qij,wqkl->wijkl",V[0,:],PV,optimize='optimal')
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
    print("*****    Solving Fock    *****")
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
    # temp  = np.zeros(VQ.shape,dtype=np.complex128)  
    VQ_mo  = np.zeros(VQ.shape,dtype=np.complex128)  
    for ik in range(VQ.shape[0]):
        for iQ in range(VQ.shape[1]):
            # Step 1: temp = M_ao[ik, iQ] @ C  -> shape (nao, nmo)
            temp = VQ[ik, iQ] @ mo_vecs
            # Step 2: M_mo[ik, iQ] = C.T @ temp -> shape (nmo, nmo)
            VQ_mo[ik, iQ] = mo_vecs.T @ temp 
    
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
    # temp4P = np.zeros((occ,virt,occ,virt))
    
    for i in range(0,occ):  
        for m in range(0,nao):  
            temp[i,:,:,:] += moVex_occ.T[i,m]*mat4P[m,:,:,:]  
        for a in range(0,virt):  
            for n in range(0,nao):  
                # temp2[i,a,:,:] += moVex_virt[a,n]*temp[i,n,:,:]  
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
    # moVex_occ  = mo_vecs[:occ,:]       # size = nao * occ
    # moVex_virt = mo_vecs[occ:nao,:]    # size = nao * virt   
    moVex_occ  = mo_vecs[:,:occ]       # size = nao * occ
    moVex_virt = mo_vecs[:,occ:]    # size = nao * virt
    
    temp  = np.zeros((occ,nao,nao,nao),dtype=np.complex128)  
    temp2 = np.zeros((occ,virt,nao,nao),dtype=np.complex128)  
    temp3 = np.zeros((occ,virt,occ,nao),dtype=np.complex128)  
    # temp4 = np.zeros((occ,virt,occ,virt),dtype=np.complex128)  
    
    mat4P_MO = np.zeros((nump,occ,virt,occ,virt),dtype=np.complex128)  
    
    # i & j are occpied orbs
    # a & b are virtual orbs
    # iajb -> mn, m = n = virt * occ
    # temp4P = np.zeros((occ,virt,occ,virt))
    for pt in range(nump):
        temp  = np.zeros((occ,nao,nao,nao),dtype=np.complex128)  
        temp2 = np.zeros((occ,virt,nao,nao),dtype=np.complex128)  
        temp3 = np.zeros((occ,virt,occ,nao),dtype=np.complex128)  
        for i in range(0,occ):  
            for m in range(0,nao):  
                temp[i,:,:,:] += moVex_occ.T[i,m]*mat4PDyn[pt,m,:,:,:]  
            for a in range(0,virt):  
                for n in range(0,nao):  
                    # temp2[i,a,:,:] += moVex_virt[a,n]*temp[i,n,:,:]  
                    temp2[i,a,:,:] += moVex_virt[n,a]*temp[i,n,:,:]  
                for j in range(0,occ):  
                    for o in range(0,nao):  
                        temp3[i,a,j,:] += moVex_occ.T[j,o]*temp2[i,a,o,:]  
                    for b in range(0,virt):  
                        for p in range(0,nao):  
                            # temp4[i,a,j,b] += moVex_virt[p,b]*temp3[i,a,j,p]  
                            mat4P_MO[pt,i,a,j,b] += moVex_virt[p,b]*temp3[i,a,j,p]  
        # mat4P_MO[pt,:] += temp4
        
    return mat4P_MO


def mo2ovDyn(mat4PDyn_mo,nelec):
    
    # nao = mat4PDyn_mo.shape[-1]
    occ  = nelec//2
    # virt = nao - occ
    # nump = mat4PDyn_mo.shape[0]

    mat4P_MO = mat4PDyn_mo[:,0:occ,occ:,0:occ,occ:]
    print(mat4P_MO.shape)
    # i & j are occpied orbs
    # a & b are virtual orbs
    # iajb -> mn, m = n = virt * occ
    # temp4P = np.zeros((occ,virt,occ,virt))

    return mat4P_MO


def mo2ovStat(mat4PStat_mo,nelec):
    occ  = nelec//2

    mat4P_MO = mat4PStat_mo[0:occ,occ:,0:occ,occ:]
    print(mat4P_MO.shape)

    return mat4P_MO


def diffEpsMat(eigVals):
    nao = len(eigVals)
    diffEps = np.zeros((nao,nao,nao,nao),dtype=np.complex128)
    # virt - occ
    for i in range(nao):
        for a in range(nao):
            diffEps[i,a,i,a] += eigVals[a] - eigVals[i]
        
    return diffEps


def concatAB(A,B):
    block_dim = A.shape[-1]
    temp = np.zeros((2*block_dim,2*block_dim),dtype=np.complex128)
    temp[0:block_dim,0:block_dim] += A
    temp[block_dim:,0:block_dim] += B
    temp[0:block_dim,block_dim:] += -B.conj()
    temp[block_dim:,block_dim:] += -A.conj()

    return temp


# def concatABDyn(A,B):
#     block_dim = A.shape[-1]
#     nump = A.shape[0]
#     temp = np.zeros((nump,2*block_dim,2*block_dim),dtype=np.complex128)
#     temp[:,0:block_dim,0:block_dim] += A
#     temp[:,block_dim:,0:block_dim] += B
#     temp[:,0:block_dim,block_dim:] += -B.conj()
#     temp[:,block_dim:,block_dim:] += -A.conj()

#     return temp


def concatABDyn(A,B):
    block_dim = A.shape[-1]
    nump = A.shape[0]
    temp = np.zeros((nump,2*block_dim,2*block_dim),dtype=np.complex128)
    temp[:,0:block_dim,0:block_dim] += A
    temp[:,block_dim:,0:block_dim] += B
    temp[:,0:block_dim,block_dim:] += -B.conj().transpose(0,2,1)
    temp[:,block_dim:,block_dim:] += -A.conj().transpose(0,2,1)

    return temp



# Fock = h5py.File("sim.h5","r")["/iter6/Fock-k"][()].view(complex)
# Sovlp = h5py.File("input.h5","r")["/HF/S-k"][()].view(complex)
# Fock = Fock.reshape(Fock.shape[:-1])
# Sovlp = Sovlp.reshape(Sovlp.shape[:-1])
# valsMO, vexMO = solveMO(Fock,Sovlp)

# print(valsMO)
# print(vexMO)

# tildeP_iw = ct.readPtilde("p_iw_tilde_q0.h5")
# VQ = ct.readVQ()
# W = getW(tildeP_iw,VQ)

def getDiffEps(valsMO,nelec):
    nao   = valsMO.shape[-1]
    occ   = nelec // 2
    virt  = nao - occ
    diffEps = diffEpsMat(valsMO[0,0,:])
    diffEps_ov = np.zeros([occ*virt,occ*virt],dtype=np.complex128)
    # i,j are in occupied, a,b are in virtual.
    for i in range(occ):
        for a in range(occ,nao):
            for j in range(occ):
                for b in range(occ,nao):
                    # iajb -> ia,jb
                    ia = i * virt + a - occ
                    jb = j * virt + b - occ
                    # matA_2D_ov[ia, jb] = matA_2D.reshape((nao,nao,nao,nao))[i, a, j, b]
                    diffEps_ov[ia, jb] = diffEps[i, a, j, b] 

    return diffEps_ov


def HDynDiagApprox(Pi,VQ,valsMO,nelec,ex_type="singlet"):
    # Must be in MO basis already.
    diffEps_ov = getDiffEps(valsMO,nelec)
    effVals, effVex = solveHstatic(Pi,VQ,diffEps_ov,nelec,ex_type)
    effVex_inv = LA.inv(effVex)
    niw  = Pi.shape[0] 
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    U1   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    U2   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
    
    # Diagonal approximation to H2p_Dyn at each frequency point.
    H2p_Dyn = np.zeros((niw,2*occ*virt),dtype=np.complex128)
    for iw in range(niw):
        print(f"Processing frequency point {iw}")
        # PV  = np.einsum("qp,pkl->qkl", Pi[iw,:], VQ[0,:,occ:,occ:], optimize='optimal') 
        PV  = np.einsum("qp,pkl->qkl", Pi[iw,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
        # ijkl: Nocc * Nocc * Nvirt * Nvirt
        VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
        # ijkl: Nocc * Nocc * Nvirt * Nvirt
        W   = VPV + U1

        A   = matEleXiStat(VQ,W,nelec,ex_type)
        A   = diffEps_ov + A.reshape(occ*virt,occ*virt)
        
        PV  = np.einsum("qp,pkl->qkl", Pi[niw//2,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
        # ijkl: Nocc * Nvirt * Nvirt * Nocc
        VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
        # ijkl: Nocc * Nvirt * Nvirt * Nocc
        W   = VPV + U2
        
        B   = matEleBStat(VQ,W,nelec,ex_type)
        B   = B.reshape(occ*virt,occ*virt)
        
        H   = concatAB(A,B)
        
        H2p_Dyn[iw] += np.einsum('ij,jk,ki->i', effVex_inv, H, effVex)
        
    return effVals, H2p_Dyn


def solveHstatic(Pi,VQ,diffEps_ov,nelec,ex_type="singlet"):
    # Must be in MO basis already.
    
    niw  = Pi.shape[0]
    occ  = nelec // 2
    nao  = VQ.shape[-1]
    virt = nao - occ
    print("*****    Solving static effective Hamiltonian    *****")
    print("shape of static Pi: ", Pi[niw//2,0,:,:,0].shape)
    print("shape of VQ fragment: ", VQ[0,:,occ:,occ:].shape)
    # Qkl: NQ * Nvirt * Nvirt
    PV  = np.einsum("qp,pkl->qkl", Pi[niw//2,0,:,:,0], VQ[0,:,occ:,occ:], optimize='optimal') 
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    print("shape of PV fragment: ", PV.shape)
    print("shape of VQ fragment: ", VQ[0,:,:occ,:occ].shape)
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,:occ], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,:occ], VQ[0,:,occ:,occ:], optimize='optimal')
    W_stat  = VPV + U
    print("shape of W_stat: ", W_stat.shape)
    # iajb: Nocc * Nvirt * Nocc * Nvirt
    A_stat = matEleXiStat(VQ,W_stat,nelec,ex_type)
    
    print("shape of static Pi: ", Pi[niw//2,0,:,:,0].shape)
    print("shape of VQ fragment: ", VQ[0,:,occ:,:occ].shape)
    # Qkl: NQ * Nvirt * Nocc
    PV  = np.einsum("qp,pkl->qkl", Pi[niw//2,0,:,:,0], VQ[0,:,occ:,:occ], optimize='optimal') 
    # ijkl: Nocc * Nvirt * Nvirt * Nocc
    print("shape of PV fragment: ", PV.shape)
    print("shape of VQ fragment: ", VQ[0,:,:occ,occ:].shape)
    VPV = np.einsum("qij,qkl->ijkl", VQ[0,:,:occ,occ:], PV, optimize='optimal')
    # ijkl: Nocc * Nocc * Nvirt * Nvirt
    U   = np.einsum('qij,qkl->ijkl', VQ[0,:,:occ,occ:], VQ[0,:,occ:,:occ], optimize='optimal')
    W_stat  = VPV + U
    print("shape of W_stat: ", W_stat.shape)
    B_stat = matEleBStat(VQ,W_stat,nelec,ex_type)
    
    A_stat  = diffEps_ov + A_stat.reshape(occ*virt,occ*virt)
    B_stat  = B_stat.reshape(occ*virt,occ*virt)
    
    H_stat  = concatAB(A_stat,B_stat)
    effVals,effVex = LA.eig(H_stat)
    
    return effVals, effVex
    
    
