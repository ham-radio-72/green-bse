import h5py
import numpy as np
import scipy.linalg as LA
from ir import IR_factory,read_IR_matrices
import gwtool as gw
import time
import math
from pyscf import gto

ir_pwd = "/home/wenm/"


def readH5(filename, dataset):
    # read .h5 file.
    fi = h5py.File(filename, 'r')  
    tensor = fi[dataset][()].view(complex)
    return tensor


def readGtau():
    # read the last iteration by default.
    # Gtau = readH5("sim.h5", "/G_tau_HF/data")
    boolDataExists = True
    i = 1
    while boolDataExists:
        try:
            Gtau_temp = readH5("sim.h5", "/iter{}/G_tau/data".format(i))
            i += 1
        except:
            boolDataExists = False
            # print("G converged after iteration {}".format(i-1))
    
    Gtau = readH5("sim.h5", "/iter{}/G_tau/data".format(i-1))
    return Gtau, (i-1)


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


def getPtilde(iter,nao,nQ,tau_h5="1e5_120.h5"):
    # Compute P tilde from G and V.
    P0 = gw.eval_P0_tilde_Q(iter, nao, nQ, tau_h5=tau_h5, meta_h5="df_hf_int/meta.h5", sim_h5="sim.h5")
    P0 = gw.symmetrize_P0(P0, tau_h5=tau_h5)
    Ptilde = gw.eval_P_tilde(P0, nQ, tau_h5=tau_h5)

    # Bosonic Fourier transformation.
    Ptau = tau2omegaFT(Ptilde)
    P = Ptau.reshape(Ptau.shape[0],1,Ptau.shape[1],Ptau.shape[2],1)

    return P


def staticUdiff(VQ, stateOption="total"):
    # \sum_Q [V_{ij,Q}V_{kl,Q} - V_{ik,Q}V_{jl,Q}]
    print("*****    diff of U    *****")
    start_time = time.time()
    nao = VQ.shape[2]
    nQ = VQ.shape[1]
    Udiff = np.zeros((nao,nao,nao,nao),dtype=np.complex128)

    V = VQ.reshape(nQ,nao,nao)  # Q,i,j
    # VV = np.einsum('qia,qbj->iabj',V,V).transpose([0,3,2,1]) # exchange (ijba)
    # VV_2 = np.einsum('qij,qba->ijba',V,V).transpose([0,1,3,2]) # coulomb (ijab)
    
    VV = np.einsum('qij,qab->ijab',V,V) # exchange (ijba)
    VV_2 = np.einsum('qia,qjb->iajb',V,V).transpose([0,2,1,3]) # coulomb (ijab)
    
    if stateOption not in ["total","sing","trip"]:
        raise Exception("Not supported excitation type.")
    if stateOption == "total":
        Udiff += (VV-VV_2)  
        # Udiff += VV
        # Udiff += (-VV-VV_2)  
    elif stateOption == "sing":
        Udiff += (2 * VV-VV_2)  
    elif stateOption == "trip":
        Udiff += (-VV_2)  
        
    print("Evaluation of U difference finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return Udiff


def getGinv(Gtau):
    # get the inverse of Gtau.
    ntau = Gtau.shape[0]
    Ginv = np.zeros(Gtau.shape,dtype=np.complex128)
    for t in range(ntau):
        Ginv[t,0,0,:,:,0] += LA.inv(Gtau[t,0,0,:,:,0])

    return Ginv


def getGinvGinv(Gtau):
    # Legacy function that computes P inv. 
    print("*****    G_inv * G_inv    *****")
    Ginv = getGinv(Gtau)
    ntau = Gtau.shape[0]
    nao = Gtau.shape[3]
    tauGinvGinv = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for t in range(ntau):
        for i in range(nao):
            for j in range(nao):
                tauGinvGinv[t,i,j,i,j] += Ginv[ntau-t-1,0,0,j,i,0] * Ginv[t,0,0,j,i,0]

    return tauGinvGinv


def getXi(ir_pwd, beta=1000, stateOption="total" ,readP = False):
    """
    Xi = U - W
    This operation is on the imaginary frequecy. 
    """
    V = readVQ()
    nao = V.shape[2]
    if readP:
        P = readPtilde("p_iw_tilde_q0.h5")
        print(P.shape)
    else:
        P = getPtilde(iter, V.shape[2], V.shape[1])

    niw = P.shape[0]
    Xi = -getVPV(V,P)
    Udiff = staticUdiff(V, stateOption)
    # if stateOption=="total":
    #     Udiff = staticUdiff(V, "total")
    # if stateOption=="sing":
    #     Udiff = staticUdiff(V, "sing")
    # if stateOption=="trip":
    #     Udiff = staticUdiff(V, "trip")
    
    for iw in range(niw):
        Xi[iw,:,:,:,:] += Udiff
    
    # # Fourier transform Xi to the imaginary time axis.
    # Xi = omega2tauFT(Xi, beta, ir_pwd)
    
    return Xi


def getXiStat(ir_pwd, beta=1000, stateOption="total" ,readP = False):
    """
    Xi = U - W (i\omega = 0)
    This operation is on the imaginary frequecy. 
    """
    V = readVQ()
    nao = V.shape[2]
    if readP:
        P = readPtilde("p_iw_tilde_q0.h5")
        print(P.shape)
    else:
        P = getPtilde(iter, V.shape[2], V.shape[1])

    niw = P.shape[0]
    middle_iw = niw // 2
    Xi = -getVPV(V,P)[middle_iw,:]
    Udiff = staticUdiff(V, stateOption)
    
    for iw in range(niw):
        Xi += Udiff

    return Xi


# def updateChi(Chi, Xi, Pi):
#     """
#     Update the Chi matrix via BSE:
#     Chi(new) = Pi + Pi Xi Chi 
#     this operation is on the imaginary frequency
#     """
#     niw = Chi.shape[0]
#     Chi_copy = np.zeros(Chi.shape,dtype=np.complex128)
#     for iw in range(niw):
#         Chi_copy[iw,:,:,:,:] = Pi[iw,:,:,:,:] + \
#                          np.einsum("ijqp,pqkl->ijkl",Pi[iw,:,:,:,:],
#                                    np.einsum("pqnm,mnkl->pqkl",Xi[iw,:,:,:,:],Chi[iw,:,:,:,:]))

#     return Chi_copy


def updateChi(Chi, Xi, Pi):
    """
    Update the Chi matrix via BSE:
    Chi(new) = Pi + Pi Xi Chi 
    this operation is on the imaginary frequency
    """
    niw = Chi.shape[0]
    Chi_copy = np.zeros(Chi.shape,dtype=np.complex128)
    if Xi.shape == 4:
        # Xi is static.
        for iw in range(niw):
            Chi_copy[iw,:,:,:,:] = Pi[iw,:,:,:,:] + \
                        np.einsum("ijqp,pqkl->ijkl",Pi[iw,:,:,:,:],
                                np.einsum("pqnm,mnkl->pqkl",Xi,Chi[iw,:,:,:,:]))
        
    for iw in range(niw):
        Chi_copy[iw,:,:,:,:] = Pi[iw,:,:,:,:] + \
                         np.einsum("ijqp,pqkl->ijkl",Pi[iw,:,:,:,:],
                                   np.einsum("pqnm,mnkl->pqkl",Xi[iw,:,:,:,:],Chi[iw,:,:,:,:]))

    return Chi_copy


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
    print(P.shape)
    print(V.shape)
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
        VPV[io,:,:,:,:] += VV.transpose([0,2,1,3])
    return VPV


def getChiinv(readP = False):
    # Calculate P by default, otherwise read P from the h5 files generated by Gaurav's UGF2 build.
    start_time = time.time()

    V = readVQ()
    G,iter = readGtau()
    if readP:
        P = readPtilde()
    else:
        P = getPtilde(iter, V.shape[2], V.shape[1])

    Chiinv = getVPV(V,P)
    Chiinv += getGinvGinv(G)
    Udiff = staticUdiff(V)
    print("*****    Summation    *****")
    for io in range(Chiinv.shape[0]):
        Chiinv[io,:,:,:,:] += Udiff

    print("Evaluation of Chi_inv finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    return Chiinv


def getChi(Chiinv):
    # Only invert the "diagonal" elements (i=k, j=l)
    # Legacy function that inverts twice.
    start_time = time.time()

    Chiinv = tau2omegaFTforG(Chiinv)
    nao = Chiinv.shape[1]
    nomega = Chiinv.shape[0]
    Chi = np.zeros(Chiinv.shape,dtype=np.complex128)
    print("*****    Inversion    *****")
    for io in range(nomega):
        for i in range(nao):
            for j in range(nao):
                Chi[io,i,j,i,j] = (1+0j) / Chiinv[io,i,j,i,j] 
    
    print("Evaluation of Chi finished.")
    print("--- %s seconds ---" % (time.time() - start_time))
    return Chi


def getPi(Gtau):
    # Get Pi on tau grid
    print("*****    Pi    *****")
    start_time = time.time()

    ntau = Gtau.shape[0]
    nao = Gtau.shape[3]
    Gtau_copy = Gtau.reshape(ntau,nao,nao)
    tauGG = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for t in range(ntau):
        # tauGG[t,:,:,:,:] = -np.einsum("ac,db->abcd",Gtau_copy[t,:,:],Gtau_copy[ntau-t-1,:,:])
        tauGG[t,:,:,:,:] = -np.einsum("da,bc->abcd",Gtau_copy[t,:,:],Gtau_copy[ntau-t-1,:,:])
    # for t in range(ntau):
    #     for i in range(nao):
    #         for j in range(nao):
    #             for k in range(nao):
    #                 for l in range(nao):
    #                     tauGG[t,i,j,k,l] -= Gtau[ntau-t-1,0,0,j,k,0] * Gtau[t,0,0,l,i,0]
                        # tauGG[t,i,j,k,l] -= Gtau[t,0,0,j,k,0] * Gtau[t,0,0,i,l,0]
                        # tauGG[t,i,j,k,l] += -1j * Gtau[t,0,0,i,k,0] * Gtau[t,0,0,l,j,0]

    print("Evaluation of Pi finished.")
    print("--- %s seconds ---" % (time.time() - start_time))

    return tauGG


# New iteration loop.
# Chi = Pi + Pi Udiff Chi - Pi tildeW Chi

def updatePiUdiffChi(Pi,Udiff,Chi):
    """
    Operating on imag freq.
    """
    nfreq = Pi.shape[0]
    PUC = np.zeros(Pi.shape,dtype=np.complex128)
    for iw in range(nfreq):
        PUC[iw,:,:,:,:] = np.einsum("ijqp,pqkl->ijkl",Pi[iw,:,:,:,:],
                                    np.einsum("pqnm,mnkl->pqkl",Udiff[:,:,:,:],Chi[iw,:,:,:,:]))
        # PUC[iw,:,:,:,:] = np.einsum("ijpq,pqkl->ijkl",Pi[iw,:,:,:,:],
        #                             np.einsum("pqmn,mnkl->pqkl",Udiff[:,:,:,:],Chi[iw,:,:,:,:]))
        
    return PUC


def updatePiWChi(Pi,tildeW,Chi):
    """
    Operating on imag time or freq
    """
    npts = Pi.shape[0]
    PWC = np.zeros(Pi.shape,dtype=np.complex128)
    if len(tildeW.shape) == 4:
        for t in range(npts):
            PWC[t,:,:,:,:] = np.einsum("ijqp,pqkl->ijkl",Pi[t,:,:,:,:],
                                    np.einsum("pqnm,mnkl->pqkl",tildeW,Chi[t,:,:,:,:]))
    else:
        for t in range(npts):
            PWC[t,:,:,:,:] = np.einsum("ijqp,pqkl->ijkl",Pi[t,:,:,:,:],
                                    np.einsum("pqnm,mnkl->pqkl",tildeW[t,:,:,:,:],Chi[t,:,:,:,:]))
        # PWC[t,:,:,:,:] = np.einsum("ijpq,pqkl->ijkl",Pi[t,:,:,:,:],
        #                             np.einsum("pqmn,mnkl->pqkl",tildeW[t,:,:,:,:],Chi[t,:,:,:,:]))
        
    return PWC


def updatePiWChi_stat(Pi,statW,Chi):
    """
    Operating on imag time or freq
    """
    npts = Pi.shape[0]
    PWC = np.zeros(Pi.shape,dtype=np.complex128)

    for t in range(npts):
        PWC[t,:,:,:,:] = np.einsum("ijqp,pqkl->ijkl",Pi[t,:,:,:,:],
                                np.einsum("pqnm,mnkl->pqkl",statW,Chi[t,:,:,:,:]))

    return PWC


def loopChi(Chi_tau,Pi_tau,Udiff,tildeW,beta,ir_h5):
    """
    operating on the imaginary freq
    """
    
    # Pi_iw  = tau2omegaFT(Pi_tau,beta,ir_h5)
    # Chi_iw = tau2omegaFT(Chi_tau,beta,ir_h5)
    
    # on imag freq
    # PUC_iw  = updatePiUdiffChi(Pi_iw, Udiff, Chi_iw)
    # PUC_tau = omega2tauFT(PUC_iw,beta,ir_h5)
    
    # on imag time
    PUC_tau = updatePiUdiffChi(Pi_tau, Udiff, Chi_tau)
    # PWC_tau = updatePiWChi(Pi_tau, tildeW, Chi_tau)
    PWC_tau = updatePiWChi_stat(Pi_tau, tildeW, Chi_tau)

    Chi_tau = Pi_tau + PUC_tau - PWC_tau
    # Chi_tau = Pi_tau - PWC_tau

    return Chi_tau


def loopChi_iw(Chi_iw,Pi_iw,Udiff,tildeW_iw,beta,ir_h5):
    """
    operating on the imaginary freq
    """

    PUC_iw  = updatePiUdiffChi(Pi_iw, Udiff, Chi_iw)
    PWC_iw = updatePiWChi(Pi_iw,tildeW_iw,Chi_iw)
    
    # Chi_iw = Pi_iw + PUC_iw - PWC_iw
    Chi_iw = Pi_iw + PUC_iw 

    return Chi_iw


def getPiQQ(Gtau,VQ):
    
    print("*****    Pi    *****")
    start_time = time.time()

    ntau = Gtau.shape[0]
    nao = Gtau.shape[3]
    Gtau_copy = Gtau.reshape(ntau,nao,nao)
    GG = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for t in range(ntau):
        # tauGG[t,:,:,:,:] = -np.einsum("ac,db->abcd",Gtau_copy[t,:,:],Gtau_copy[ntau-t-1,:,:])
        GG[t,:,:,:,:] = -np.einsum("da,bc->abcd",Gtau_copy[t,:,:],Gtau_copy[ntau-t-1,:,:])
        
    GG = getPi(Gtau)
    for t in range(ntau):
        PiQQ = np.einsum("abcd,qab->qcd",GG[t,:,:,:,:],VQ)



# ===================================================== #

def getPiinv(Pi):
    # Legacy function that computes Pi inv. 
    ntau = Pi.shape[0]
    nao = Pi.shape[1]

    Piinv = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)
    for it in range(ntau):
        Piinv[it,:,:,:,:] += LA.inv(Pi[it,:,:,:,:].reshape(nao*nao,nao*nao)).reshape(nao,nao,nao,nao)
    
    return Piinv


def getI_PiXi(Pi):

    V = readVQ()
    nao = V.shape[2]
    if readP:
        P = readPtilde()
    else:
        P = getPtilde(iter, V.shape[2], V.shape[1])

    ntau = P.shape[0]
    Xi = getXi()

    I_PiXi = np.zeros((ntau,nao,nao,nao,nao),dtype=np.complex128)

    print("*****    I - PiXi    *****")
    start_time = time.time()
    for it in range(ntau):
        identity = np.einsum('ia,jb->ijba',np.eye(nao,dtype=np.complex128),np.eye(nao,dtype=np.complex128))
        I_PiXi[it,:,:,:,:] += identity - np.einsum("ijnm,mnba",Xi[it,:],Pi[it,:])

    print("Evaluation of (I - PiXi) finished.")    
    print("--- %s seconds ---" % (time.time() - start_time))
    return I_PiXi


def getChi_alt(I_PiKi,Pi):
    print("*****    Chi    *****")
    # Using the Chi = (I - Pi * Xi)^(-1) * Pi scheme.
    start_time = time.time()

    ntau = Pi.shape[0]
    nao = Pi.shape[1]
    Chi = np.zeros(Pi.shape,dtype=np.complex128)
    identity = np.einsum('ik,jl->ijkl',np.eye(nao,dtype=np.complex128),np.eye(nao,dtype=np.complex128))
    
    for it in range(ntau):
        # Inverting the flatten version of I_PiKi.
        # For now the off-diagonal elements are not ignored. 
        I_PiKiinv_2D = LA.inv(I_PiKi[it,:,:,:,:].reshape(nao*nao,nao*nao))
        I_PiKiinv_2D = np.einsum('ab,bc->ac',I_PiKiinv_2D,identity.reshape(nao*nao,nao*nao))
        I_PiKiinv = I_PiKiinv_2D.reshape(nao,nao,nao,nao)
        # I_PiKiinv = LA.inv(I_PiKi[it,:,:,:,:].reshape(nao*nao,nao*nao)).reshape(nao,nao,nao,nao)
        # Chi[it,:,:,:,:] += np.matmul(I_PiKiinv_2D,Pi[it,:,:,:,:].reshape(nao*nao,nao*nao)).reshape(nao,nao,nao,nao)
        # Chi[it,:,:,:,:] += np.einsum('ijpq,qpkl->ijkl',I_PiKiinv,Pi[it,:,:,:,:])
        Chi[it,:,:,:,:] += np.einsum('ijpq,pqkl->ijkl',I_PiKiinv,Pi[it,:,:,:,:])
    print("Evaluation of Chi finished.")    
    print("--- %s seconds ---" % (time.time() - start_time))
    return Chi


def getUChi(Chi,beta,ir_h5):
    # For molecular microscopic epsilon.
    print("*****    Macroscopic dielectric    *****")
    start_time = time.time()
    ntau = Chi.shape[0]
    nao = Chi.shape[1]
    VQ = readVQ()
    nQ = VQ.shape[1]
    V = VQ.reshape(nQ,nao,nao)  # Q,i,j
    
    U = np.einsum('qab,qcd->abcd',V,V)
    UChi_ijkl = np.einsum("ijnm,tmnkl->tijkl",U,Chi)
    # UChi_ijkl = np.einsum("imnl,tmjkn->tijkl",U,Chi)
    UChi2p_il = np.einsum("tijkl->til",UChi_ijkl)
    
    print("F-T UChi to imag frequency...")
    UChi2p_iomega = tau2omegaFT(UChi2p_il,beta,ir_h5)
    print("Evaluation of macroscopic dielectric finished.")    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    return UChi2p_iomega,UChi2p_il


# def getUChi(Chi,beta,ir_h5):
#     # For molecular microscopic epsilon.
#     print("*****    Macroscopic dielectric    *****")
#     start_time = time.time()
#     ntau = Chi.shape[0]
#     nao = Chi.shape[1]
#     VQ = readVQ()
#     nQ = VQ.shape[1]
#     V = VQ.reshape(nQ,nao,nao)  # Q,i,j
    
#     U = np.einsum('qab,qcd->abcd',V,V)
#     U = np.einsum('abcd->ad',U)
#     Chi2p_il  = np.einsum("tijkl->til",Chi)
#     UChi2p_il = np.einsum("ia,tal->til",U,Chi2p_il)
    
#     print("F-T UChi to imag frequency...")
#     UChi2p_iomega = tau2omegaFT(UChi2p_il,beta,ir_h5)
#     print("Evaluation of macroscopic dielectric finished.")    
#     print("--- %s seconds ---" % (time.time() - start_time))
    
#     return UChi2p_iomega,UChi2p_il


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

    # Chi_2D = np.zeros((Chi.shape[0],Chi.shape[1],Chi.shape[2],nao,nao),dtype=np.complex128)
    # Chi_ortho = np.zeros(Chi.shape,dtype=np.complex128)
    # print(Chi_2D.shape)
    # for i in range(nao):
    #     for j in range(nao):
    #         Chi_2D[:,:,:,i,j] += Chi[:,:,:,i,j,i,j,0]
    # del Chi
    # Chi_ortho_2D = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, Chi_2D, s_c, optimize=True)
    # del Chi_2D
    # for i in range(nao):
    #     for j in range(nao):
    #         Chi_ortho[:,:,:,i,j,i,j,0] += Chi_ortho_2D[:,:,:,i,j,]

    # return Chi_ortho,Chi_ortho_2D



def orth_mo_2p(tensor_2p,Fock,S_ovlp):
    """
    Orthogonalizing matrix from AO to MO.
    """
    # Solve for MO eigenvectors.
    # Solve the generalized eigen problem: FC = SCE
    print(tensor_2p.shape)
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


def getMicroEps(UChi,Fock,S_ovlp,orthBool=False):
    """
    Input UChi (un-orthogonalized) on the iomega axis.
    Option to orthogonalize in the MO basis.
    """
    nao = UChi.shape[-1]
    nfreq = UChi.shape[0]
    
    if orthBool:
        UChi = UChi.reshape(nfreq,1,1,nao,nao)
        UChiOrth = orth_mo_2p(UChi,Fock,S_ovlp)
        UChiOrth = UChiOrth.reshape(nfreq,nao,nao)
        UChi = UChiOrth

    microEps = np.zeros(UChi.shape,dtype=np.complex128)
    # microEps = UChi + 1.0
    for io in range(nfreq):
        microEps[io,:,:] += UChi[io,:,:] + np.eye(nao)
        # microEps[io,:,:] += UChi[io,:,:] + 1.0
        # UChi[io,:,:] += np.eye(nao,dtype=np.complex128) 
        # UChi[io,:,:] += np.eye(nao) 
    
    return microEps


def getMacroEps(microEps):
    """
    Input UChi (un-orthogonalized) on the iomega axis.
    Option to orthogonalize in the MO basis.
    """
    
    print("Inverting microEps to macroEps...")
    nfreq = microEps.shape[0]
    nao = microEps.shape[-1]
    macroEps = np.zeros(microEps.shape,dtype=np.complex128)
    for io in range(nfreq):
        macroEps[io,:,:] += LA.inv(microEps[io,:,:])
    # for io in range(nfreq):
    #     for i in range(nao):
    #         for j in range(nao):
    #             macroEps[io,i,j] += 1/microEps[io,i,j]
    return macroEps


def getEpsMinusDelta(macroEps,Fock,S_ovlp,beta,ir_h5,orthBool=True):
    """
    Input macro epsilon to get the difference with delta.
    """
    nfreq = macroEps.shape[0]
    nao   = macroEps.shape[-1]
    macroEps_temp = macroEps.copy()
    for io in range(nfreq):
        macroEps_temp[io,:,:] = macroEps[io,:,:] - np.eye(nao)
    
    # if orthBool:
    #     macroEps_temp = macroEps_temp.reshape(nfreq,1,1,nao,nao)
    #     epsMinusDelta = orth_mo_2p(macroEps_temp,Fock,S_ovlp)
    #     epsMinusDelta = epsMinusDelta.reshape(nfreq,nao,nao)
    # else:
    #     epsMinusDelta = macroEps_temp
        
    # epsMinusDelta_tau = omega2tauFTforG(macroEps_temp,beta,ir_h5)
    epsMinusDelta_tau = omega2tauFT(macroEps_temp,beta,ir_h5)
    
    ntau = epsMinusDelta_tau.shape[0]
    epsMinusDelta_tau_temp = epsMinusDelta_tau.copy()
    if orthBool:
        epsMinusDelta_tau_temp = epsMinusDelta_tau_temp.reshape(ntau,1,1,nao,nao)
        epsMinusDelta_tau_temp = orth_mo_2p(epsMinusDelta_tau_temp,Fock,S_ovlp)
        epsMinusDelta_tau_temp = epsMinusDelta_tau_temp.reshape(ntau,nao,nao)
        
    return macroEps_temp, epsMinusDelta_tau_temp


def getMicroEpsSum(UChi):
    """ Sum UChi over all space and add 1 (as the vaccum dielectric). """
    microEps = np.zeros(UChi.shape,dtype=np.complex128)
    # summation over all space
    microEps = np.einsum("wij->w",UChi)
    microEps += 1.0

    return microEps


def getMacroEpsSum(microEps):
    macroEps = np.zeros(microEps.shape,dtype=np.complex128)
    nfreq = microEps.shape[0]
    for io in range(nfreq):
        macroEps[io] = 1/microEps[io]
        
    return macroEps
    
    
def contractChi2p(Chi):
    """
    Contracting four-point Chi to two-point 
    """
    ntau = Chi.shape[0]
    nao = Chi.shape[1]
    print("Contracting Chi to two points...")
    Chi2p = np.zeros((ntau,nao,nao),dtype=np.complex128)
    for it in range(ntau):
        # Chi2p[it,:,:] += np.einsum("iikk->ik",Chi[it,:,:,:,:])
        Chi2p[it,:,:] += np.einsum("ijkj->ik",Chi[it,:,:,:,:])

    return Chi2p


def getVext(mol_coords, basis):
    """
    Calculate the external potential given by the atomic coordinates.
    Args:
        mol: pyscf object
    """
    mol = gto.M(atom=mol_coords)
    mol.basis = basis
    mol.build()
    vext = mol.intor_symmetric('int1e_nuc')
    
    return vext


def getVChiV(Chi2p, Vext):
    ntau = Chi2p.shape[0]
    VChi2pV = np.zeros(Chi2p.shape,dtype=np.complex128)
    for it in range(ntau):
        VChi2pV[it,:,:] = np.einsum("ai,ij,jb->ab",Vext,Chi2p[it,:,:],Vext)
     
    return VChi2pV

