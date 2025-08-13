import sys 
sys.path.append('/home/wenm/green-bse/src')

import casidaEq as casida
import h5py
import contract as ct
import numpy as np
import scipy.linalg as LA
import argparse
import qp
import time


# Read arguments.
parser = argparse.ArgumentParser(
    description="Nevanlinna analytic continuation for n-th iteration: Uses the updated Fock"
)
parser.add_argument(
    "--beta", type=float, default=1000, help="Inverse temperature"
)
parser.add_argument(
    "--sim", type=str, default="sim.h5",
    help="Output of scGW"
)
parser.add_argument(
    "--input", type=str, default="input.h5",
    help="Input of scGW"
)
parser.add_argument(
    "--int_path", type=str, default="df_hf_int/",
    help="Path for the VQ.h5 file"
)
parser.add_argument(
    "--output", type=str, default="bse.h5",
    help="Output of BSE"
)
parser.add_argument(
    "--iter", type=int, default=-1,
    help="Iteration number of the scGW cycle to use for continuation"
)
parser.add_argument(
    "--ir_file", type=str, default=None,
    help="HDF5 file that contains information about the IR grid."
)
parser.add_argument(
    "--pi_file", type=str, default="p_iw_tilde_q0.h5",
    help="HDF5 file that contains information about non-interacting susceptibility."
)
parser.add_argument(
    "--calc_pi", type=bool, default=False,
    help="Option to caluclate Pi on the fly. Not implemented yet."
)
parser.add_argument(
    "--type", type=str, default="normal",
    help="Type of excitations."
)
parser.add_argument(
    "--ns", type=int, default=1,
    help="Number of spins"
)
parser.add_argument(
    "--qpac", type=int, default=False,
    help="Enable QP AC?"
)

# Parse arguments
args = parser.parse_args()

input_path = args.input
int_path = args.int_path
sim_path = args.sim
out_path = args.output
beta = args.beta
ex_type = args.type
ns = args.ns
qpac_bool = args.qpac

it = args.iter
ir_file = args.ir_file
pi_file = args.pi_file


print("*****    Starting Casida solver for scGW iteration: ", it, "    *****")

start = time.time()
print("Reading IR file")
wgrid = h5py.File(ir_file,"r")["/bose/wsample"][()]
wgrid = 2 * wgrid * np.pi / beta

print("Reading sim file")
f = h5py.File(input_path, 'r')
rSk = f["/HF/S-k"][()].view(complex)
rSk = rSk.reshape(rSk.shape[:-1])
nao   = f["/params/nao"][()]
nelec = f["/params/nel_cell"][()]
occ   = nelec // 2
virt  = nao - occ

print("Reading sim file")
f = h5py.File(sim_path, 'r')
if it == -1:
    it = f["iter"][()]
rFk = f["iter" + str(it) + "/Fock-k"][()].view(complex)
rFk = rFk.reshape(rFk.shape[:-1])
rSigmak = f["iter" + str(it) + "/Selfenergy/data"][()].view(complex)
rSigmak = rSigmak.reshape(rSigmak.shape[:-1])
mu = f["iter" + str(it) + "/mu"][()]

end = time.time()
print("*****    Time taken to read files: ", end - start, " seconds.    *****")


start = time.time()
# Solve for eigenvals from the Fock matrix.
print("*****    Solving for MO eigenvalues    *****")
valsMO, vexMO = casida.solveMO(rFk,rSk)
print("The dimension for vexMO: ", vexMO.shape)
mo_coeff_gamma = np.zeros((ns,1,nao,nao))
mo_coeff_gamma[0,0,:,:] = vexMO[...]
mo_coeff = 1.0 * mo_coeff_gamma
mo_coeff_adj = np.einsum('skpq -> skqp', mo_coeff.conj())

# Pade interpolation using Sigma
# return quasiparticle energy
# transform Sigma from AO to MO with the updated fock mo coeff
if qpac_bool:
    print("Getting QP ac energy levels from self-energy.")
    Sigma_tk_int = np.einsum(
        'skab, tskbc, skcd -> tskad',
        mo_coeff_adj, rSigmak, mo_coeff,
        optimize=True
    )
    print(Sigma_tk_int.shape)
    valsMO_qp = qp.padeSigma(Sigma_tk_int,valsMO,beta,mu,ir_file)

end = time.time()
print("*****    Time taken to solve MO eigenvalues: ", end - start, " seconds.    *****")

start = time.time()
tildeP_iw = ct.readPtilde(pi_file)
VQ = ct.readVQ(int_path + "VQ_0.h5")
VQ = casida.VQ_ao2mo(VQ,vexMO[0,0,:])  # Orthogonalize VQ to MO basis
print("Shape of VQ after ao2mo: ", VQ.shape)
if qpac_bool:
    valsMO = valsMO_qp

effVals, H2p_Dyn = casida.HDynDiagApprox(tildeP_iw,VQ,valsMO,nelec,ex_type)
niw = H2p_Dyn.shape[0]

end = time.time()
print("*****    Time taken to solve the effective two-particle Hamiltonian: ", end - start, " seconds.    *****")   


# start = time.time()
# print("*****    Preparing U and W matrices    *****")
# tildeP_iw = ct.readPtilde(pi_file)
# VQ = ct.readVQ(int_path + "VQ_0.h5")
# VQ = casida.VQ_ao2mo(VQ,vexMO[0,0,:])  # Orthogonalize VQ to MO basis

# W = casida.getW(tildeP_iw,VQ)
# niw = W.shape[0]
# end = time.time()
# print("*****    Time taken to prepare W: ", end - start, " seconds.    *****")
# # # examine efficiency of einsum.
# # start = time.time()
# # U = np.einsum('qij,qkl->ijkl',VQ[0,:],VQ[0,:])
# # # statW = W[niw//2,:]
# # end = time.time()
# # print("*****    Time taken to prepare U: ", end - start, " seconds.    *****")

# print("*****    Preparing the dynamical Hamiltonian    *****")
# start = time.time()
# Xi_Dyn = casida.matEleXiDyn(VQ,W,ex_type)
# B_Dyn  = casida.matEleBDyn(VQ,W,ex_type)
# end = time.time()
# print("*****    Time taken to prepare Xi and B: ", end - start, " seconds.    *****")

# # time transformation of Xi_Dyn and B_Dyn from AO to OV.
# start = time.time()
# # Xi_Dyn_OV = casida.ao2ovDyn(Xi_Dyn,rFk,rSk,nelec)
# # B_Dyn_OV = casida.ao2ovDyn(B_Dyn,rFk,rSk,nelec)

# Xi_Dyn_OV = casida.mo2ovDyn(Xi_Dyn,nelec)
# B_Dyn_OV = casida.mo2ovDyn(B_Dyn,nelec)

# end = time.time()
# print("*****    Time taken to transform Xi and B to OV space: ", end - start, " seconds.    *****")
# start = time.time()
# Xi_Dyn_OV = Xi_Dyn_OV.reshape([niw,occ*virt,occ*virt])
# B_Dyn_OV = B_Dyn_OV.reshape([niw,occ*virt,occ*virt])
# end = time.time()
# print("*****    Time taken to reshape Xi and B: ", end - start, " seconds.    *****")

# start = time.time()
# diffEps = casida.diffEpsMat(valsMO[0,0,:])
# if qpac_bool:
#     diffEps = casida.diffEpsMat(valsMO_qp[0,0,:])  # QP energy level

# diffEps_ov = np.zeros([occ*virt,occ*virt],dtype=np.complex128)
# # i,j are in occupied, a,b are in virtual.
# for i in range(occ):
#     for a in range(occ,nao):
#         for j in range(occ):
#             for b in range(occ,nao):
#                 # iajb -> ia,jb
#                 ia = i * virt + a - occ
#                 jb = j * virt + b - occ
#                 # matA_2D_ov[ia, jb] = matA_2D.reshape((nao,nao,nao,nao))[i, a, j, b]
#                 diffEps_ov[ia, jb] = diffEps[i, a, j, b] 

# A_Dyn_OV = np.zeros(Xi_Dyn_OV.shape,dtype=np.complex128)
# for iw in range(niw):
#     A_Dyn_OV[iw,:] = Xi_Dyn_OV[iw,:] + diffEps_ov 

# H2p_Dyn = casida.concatABDyn(A_Dyn_OV,B_Dyn_OV)
# end = time.time()
# print("*****    Time taken to prepare the dynamical Hamiltonian: ", end - start, " seconds.    *****")  

# print("*****    Solving the effective two-particle Hamiltonian    *****")
# start = time.time()
# effVals,effVex = LA.eig(H2p_Dyn[niw//2,:])
print("The eigenvalues solved from the static effective two-particle Hamiltonian: ")
print(effVals.real)
# end = time.time()
# print("*****    Time taken to solve the effective two-particle Hamiltonian: ", end - start, " seconds.    *****")   

# Construct the auxiliary function.
print("*****    Writing output    *****")
start = time.time()
F2p_Dyn = H2p_Dyn.copy()
imag_iw = np.zeros((niw,2*occ*virt),dtype=np.complex128)
for iw in range(niw):
    imag_iw[iw,:] += wgrid[iw] * 1j
    # imag_iw[iw,:] -= wgrid[iw] * 1j
F2p_Dyn += imag_iw

for iw in range(niw):
    F2p_Dyn[iw,:] = 1/F2p_Dyn[iw,:]

F2p_Dyn_tau = ct.omega2tauFT(F2p_Dyn,beta,ir_file)


f = h5py.File(out_path, 'w')
f["/ResFunc_tau/data"]  = F2p_Dyn_tau
f["/ResFunc_iw/data"]   = F2p_Dyn
f["/StatExcVals/data"]  = effVals
# f["/FockValDiffs/data"] = diffEps_ov
f["/MOvectors/data"]    = vexMO
# f["/StatExcVex/data"]   = effVex
f["/effHamil/data"]     = H2p_Dyn[niw//2,:]
# f["/QPValDiffs/data"] = diffEps_ov
f.close()

end = time.time()
print("*****    Time taken to write output: ", end - start, " seconds.    *****")


