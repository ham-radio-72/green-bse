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


def matEleXiDyn(VQ,W):
    # A(ia,jb) = Eps(ia,jb) + U(ia,jb) - W(ij,ab)
    U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    A = np.zeros(W.shape,dtype=np.complex128)
    nump = W.shape[0]
    for p in range(nump):
        W_temp = W[p,:]
        A[p,:] += U - W_temp.transpose([0,2,1,3])
    
    return A


def matEleBDyn(VQ,W):
    # B(ia,jb) = Eps(ia,bj) - W(ib,aj)
    U = np.einsum('qia,qjb->iajb',VQ[0,:],VQ[0,:])
    B = np.zeros(W.shape,dtype=np.complex128)
    nump = W.shape[0]
    nao  = W.shape[-1]
    # for p in range(nump):
    #     W_temp = W[p,:]
    #     B[p,:] += U.transpose([0,1,3,2]) - W_temp.transpose([0,3,1,2])
    for p in range(nump):
        for a in range(nao):
            for j in range(nao):
                for b in range(nao):
                    B[p,:,a,j,b] = U[:,a,b,j] - W[p,:,b,a,j]
        
    return B


def getW(P,V):
    print("*****    VPV    *****")
    print(P.shape)
    print(V.shape)
    
    pts = P.shape[0]
    PV = np.einsum("wqp,pkl->wqkl", P[:,0,:,:,0],V[0,:])
    W = np.einsum("qij,wqkl->wijkl",V[0,:],PV)
    U = np.einsum('qij,qkl->ijkl',V[0,:],V[0,:])
    
    print("*****    U    *****")
    for p in range(pts):
        W[p,:] = W[p,:] + U

    print(W.shape)
    
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


def ao2mo(mat4P,Fock,S_ovlp):
    
    nao = mat4P.shape[-1]
    _,mo_vecs = solveMO(Fock,S_ovlp)
    mo_vecs = mo_vecs.reshape(nao,nao).T
    
    dim   = mat4P.shape[0]
    temp  = np.zeros(mat4P.shape,dtype=np.complex128)  
    temp2 = np.zeros(mat4P.shape,dtype=np.complex128)  
    temp3 = np.zeros(mat4P.shape,dtype=np.complex128)  
    mat4P_MO = np.zeros(mat4P.shape,dtype=np.complex128)  
    for i in range(0,dim):  
        for m in range(0,dim):  
            temp[i,:,:,:] += mo_vecs[i,m]*mat4P[m,:,:,:]  
        for j in range(0,dim):  
            for n in range(0,dim):  
                temp2[i,j,:,:] += mo_vecs[j,n]*temp[i,n,:,:]  
            for k in range(0,dim):  
                for o in range(0,dim):  
                    temp3[i,j,k,:] += mo_vecs[k,o]*temp2[i,j,o,:]  
                for l in range(0,dim):  
                    for p in range(0,dim):  
                        mat4P_MO[i,j,k,l] += mo_vecs[l,p]*temp3[i,j,k,p]  

    return mat4P_MO


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


def concatABDyn(A,B):
    block_dim = A.shape[-1]
    nump = A.shape[0]
    temp = np.zeros((nump,2*block_dim,2*block_dim),dtype=np.complex128)
    temp[:,0:block_dim,0:block_dim] += A
    temp[:,block_dim:,0:block_dim] += B
    temp[:,0:block_dim,block_dim:] += -B.conj()
    temp[:,block_dim:,block_dim:] += -A.conj()

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



