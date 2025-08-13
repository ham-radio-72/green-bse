import scipy.linalg as LA
import numpy as np


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

    return eiv_sk, mo_coeff_sk


def orth_mo_4p(Chi,Fock,S_ovlp):
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
    nao = Chi.shape[-1]
    Chi_2D = np.zeros((Chi.shape[0],Chi.shape[1],Chi.shape[2],nao,nao),dtype=np.complex128)
    Chi_ortho = np.zeros(Chi.shape,dtype=np.complex128)
    print(Chi_2D.shape)
    for i in range(nao):
        for j in range(nao):
            Chi_2D[:,:,:,i,j] += Chi[:,:,:,i,j,i,j,0]
    del Chi
    Chi_ortho_2D = np.einsum('skab, wskbc, skcd -> wskad', cdag_s, Chi_2D, s_c, optimize=True)
    del Chi_2D
    for i in range(nao):
        for j in range(nao):
            Chi_ortho[:,:,:,i,j,i,j,0] += Chi_ortho_2D[:,:,:,i,j,]

    return Chi_ortho,Chi_ortho_2D


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
