# 4 steps for the whole code:
# Construct the qubits, trotterized evolution, controlled evolution, quantum fourier transform
# quspin package is required to simplify the simulation of PXP model with constrained Hilbert space

from __future__ import print_function, division
import matplotlib.pyplot as plt
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d, spin_basis_general # Hilbert space spin basis_1d

from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import pre_check_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
from scipy.stats import gaussian_kde

import torch as pt
from torch import matrix_exp as expm

N_PXP = 8 # total number of lattice sites
L_qubit = 5 # number of the ancillary qubits, in the article it is set to be 3, 4, 5

# define the qubits and quantum gates
id_local = pt.tensor([[1.+ 0j, 0], [0, 1.]])
sigma_x = pt.tensor([[0 + 0j, 1.], [1., 0]])
sigma_y = 1j * pt.tensor([[0, -1.], [1 + 0j, 0]])
sigma_z = pt.tensor([[1., 0+ 0j], [0, -1.]])

hadamard = 1.0 / pt.sqrt(pt.tensor(2)) * pt.tensor([[1, 1], [1, -1]]) + 1j * 0


def get_Identity(k):  # returns k-tensor product of the identity operator, ie. Id^k
    Id = id_local
    for i in range(0, k-1):
        Id = pt.kron(Id, id_local)
    return Id

def get_chain_operator(A, L, i):
    if (i == 0):
        Op = pt.kron(A, get_Identity(L - 1))

    elif (i == L-1):
        Op = pt.kron(get_Identity(L - 1), A)

    elif (i > 0 and i < L-1):
        Op = pt.kron(get_Identity(i), pt.kron(A, get_Identity(L - i - 1)))

    return Op

def get_chain_operators(L):
    Id = get_chain_operator(id_local, L, 1)
    X = {}
    Y = {}
    Z = {}
    H = {}


    for qubit_i in range(L):  # Loop over indices on a 2-dimensional grid (i_x,i_y)
        X[qubit_i] = get_chain_operator(sigma_x, L, qubit_i)  # Define operator X_i acting on spin (i_x,i_y)
        Y[qubit_i] = get_chain_operator(sigma_y, L, qubit_i)  # Define operator Y_i acting on spin (i_x,i_y)
        Z[qubit_i] = get_chain_operator(sigma_z, L, qubit_i)  # Define operator Z_i acting on spin (i_x,i_y)
        H[qubit_i] = get_chain_operator(hadamard, L, qubit_i)
    return Id, X, Y, Z, H

I, X, Y, Z, Ha = get_chain_operators(L_qubit)

def Rx(theta, j):
    return expm(-1j * theta / 2 * X[j])

def Ry(theta, j):
    return expm(-1j * theta / 2 * Y[j])

def Rz(theta, j):
    return expm(-1j * theta / 2 * Z[j])

def CRn(i, j, n):
    Rn = expm(-1j * (2 * np.pi/ (2**(n+1)) )* Z[j]) * np.exp(1j*np.pi/(2**n))
    return 1 / 2 * ((I - Z[i]) @ (Rn) + (I + Z[i]) @ I)

def QFT_3(l, k, m):
    gate = CRn(m, l, 3) @ CRn(k, l, 2) @ Ha[l]
    gate = Ha[m] @ CRn(m, k, 2) @ Ha[k] @ gate
    return gate

def QFT_4(l, k, m, h):
    gate = QFT_3(k, m, h) @ CRn(h, l, 4) @ CRn(m, l, 3) @ CRn(k, l, 2) @ Ha[l]
    return gate

def QFT_5(b, l, k, m, h):
    gate = QFT_4(l, k, m, h) @ CRn(h, b, 5) @ CRn(m, b, 4) @ CRn(k, b, 3) @ CRn(l, b, 2) @ Ha[b]
    return gate

# Use quspin to simulate the evolution of PXP model in constrained Hilbert space
def Zn(L, n):
 if n == 0:
  return ('0'* L )
 if L%n == 0:
  return ('1'+'0'*(n-1))*(L//n)
 elif L%n - 1 == 0:
  return ('1'+'0'*(n-1))*(L//n) + ('0')
 else:
  return ('1' + '0' * (n - 1)) * (L // n) + ('1' + '0' * (L % n - 1))


@cfunc(op_sig_32, locals=dict(s=int32,b=uint32))
def op(op_struct_ptr,op_str,ind,N,args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state>>ind)&1)<<1)-1
    b = (1<<ind)
    #
    if op_str==120: # "x" is integer value 120 (check with ord("x"))
        op_struct.state ^= b
    elif op_str==121: # "y" is integer value 120 (check with ord("y"))
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j*s
    elif op_str==122: # "z" is integer value 120 (check with ord("z"))
        op_struct.matrix_ele *= s
    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err
#
op_args=np.array([],dtype=np.uint32)
#
######  function to filter states/project states out of the basis, required for PXP model
#
@cfunc(pre_check_state_sig_32,
    locals=dict(s_shift_left=uint32,s_shift_right=uint32), )
def pre_check_state(s,N,args):
    """ imposes that that a bit with 1 must be preceded and followed by 0,
    i.e. a particle on a given site must have empty neighboring sites.
    #
    Works only for lattices of up to N=32 sites (otherwise, change mask)
    #
    """
    mask = (0xffffffff >> (32 - N)) # works for lattices of up to 32 sites
    # cycle bits left by 1 periodically
    s_shift_left = (((s << 1) & mask) | ((s >> (N - 1)) & mask))
    #
    # cycle bits right by 1 periodically
    s_shift_right = (((s >> 1) & mask) | ((s << (N - 1)) & mask))
    #
    return (((s_shift_right|s_shift_left)&s))==0
#
pre_check_state_args=None
#
######  define symmetry maps
#
@cfunc(map_sig_32)
def parity(x, N, sign_ptr, args):
    """ works for all system sizes N, spin-1/2 only. """
    out = 0
    s = N - 1
    #
    out ^= (x & 1)
    x >>= 1
    while (x):
        out <<= 1
        out ^= (x & 1)
        x >>= 1
        s -= 1
    #
    out <<= s
    return out


P_args = np.array([], dtype=np.uint32)


@cfunc(map_sig_32)
def translation(x, N, sign_ptr, args):
    """ works for all system sizes N, spin-1/2 only. """
    shift = 1  # translate state by shift sites
    period = N  # periodicity/cyclicity of translation
    xmax = (1 << N) - 1  # largest integer allowed to appear in the basis
    #
    l = (shift + period) % period
    x1 = (x >> (period - l))
    x2 = ((x << l) & xmax)
    #
    return (x2 | x1)


T_args = np.array([], dtype=np.uint32)
#
######  construct user_basis, we take the space in the first symmetric block to reduce the computatinal resouce
# define maps dict
maps = dict(T_block=(translation, N_PXP, 0, T_args), P_block=(parity, N_PXP, 0, P_args), )  # no symmetries to apply.

# maps = {}
# define particle conservation and op dicts
op_dict = dict(op=op,op_args=op_args)
# define pre_check_state
pre_check_state=(pre_check_state,pre_check_state_args) # None gives a null pinter to args
# create user basis
basis = user_basis(np.uint32,N_PXP,op_dict,allowed_ops=set("xyz"),sps=2,
                    pre_check_state=pre_check_state,Ns_block_est=300000,**maps)

# Define Z2 (Neel) state in the simplified basis
z0 = basis.index(Zn(N_PXP, 0))
z2 = basis.index(Zn(N_PXP, 2))
Z2 = np.zeros(basis.Ns)


Z2[z2]=1
Z2_ = pt.from_numpy(Z2)
Z2_ = Z2_.to(pt.cfloat)
d = np.shape(Z2_)[0]

# Trottered hamiltonian evolution for PXP at delta time step
def U_t(delta, m):
    # delta: trotter steps
    # m: horizontal field
    h_list = [[1, i] for i in range(N_PXP)]
    z_list = [[m, i] for i in range(N_PXP)]

    # operator string lists
    static = [["z", z_list], ["x", h_list],]
    # compute Hamiltonian, no checks have been implemented
    no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)

    H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    H_array = H.todense()
    H_tensor = pt.from_numpy(H_array)
    H_tensor = H_tensor.to(pt.cfloat)
    return expm(-1j * H_tensor * delta), H_tensor

# To make the trottered time-evolution controlled by qubit c
def Controlled_U(t, c, m):
    # t: time
    # the controlled qubit
    # m: the horizontal field
    gate = pt.eye(d, dtype=pt.cfloat)
    # Forward and Backward
    U3_ = U_t(t, m)[0] @ gate
    U3__ = U_t(-1*t, m)[0] @ gate
    gate = pt.kron(U3_, U3__)
    return pt.kron((0.5 * (I + Z[c])), pt.eye(d**2, dtype=pt.cfloat)) +  pt.kron((0.5 * (I - Z[c])), gate)


def IPR_algorithm(m, L):
    t = 0.5

    psi_1 = pt.zeros(2 ** L_qubit, dtype=pt.cfloat)
    psi_1[0] = 1
    psi_1 = Ha[0] @ psi_1
    psi_1 = Ha[1] @ psi_1
    psi_1 = Ha[2] @ psi_1

    I0 = pt.kron(0.5 * (I + Z[0]) , pt.eye(d**2, dtype=pt.cfloat))
    I1 = pt.kron(0.5 * (I + Z[1]) , pt.eye(d**2, dtype=pt.cfloat))


    if L == 3:

        psi_hybrid_1 = pt.kron(pt.kron(psi_1, Z2_), Z2_)
        psi_hybrid_1 = Controlled_U(t, 0, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(2 * t, 1, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(4 * t, 2, m) @ psi_hybrid_1
        psi_hybrid_1 = pt.kron(QFT_3(0, 1, 2).contiguous(), pt.eye(d ** 2, dtype=pt.cfloat)) @ psi_hybrid_1
        I2 = pt.kron(0.5 * (I + Z[2]), pt.eye(d ** 2, dtype=pt.cfloat))
        ipr = pt.vdot(psi_hybrid_1, I0 @ I1 @ I2 @ psi_hybrid_1)
    if L == 4:
        psi_1 = Ha[3] @ psi_1
        psi_hybrid_1 = pt.kron(pt.kron(psi_1, Z2_), Z2_)
        psi_hybrid_1 = Controlled_U(t, 3, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(2 * t, 2, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(4 * t, 1, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(8 * t, 0, m) @ psi_hybrid_1
        psi_hybrid_1 = pt.kron(QFT_4(0, 1, 2, 3).contiguous(), pt.eye(d ** 2, dtype=pt.cfloat)) @ psi_hybrid_1
        I2 = pt.kron(0.5 * (I + Z[2]), pt.eye(d ** 2, dtype=pt.cfloat))
        I3 = pt.kron(0.5 * (I + Z[3]), pt.eye(d ** 2, dtype=pt.cfloat))
        ipr = pt.vdot(psi_hybrid_1, I0 @ I1 @ I2 @ I3 @ psi_hybrid_1)
    if L == 5:
        psi_1 = Ha[3] @ psi_1
        psi_1 = Ha[4] @ psi_1
        psi_hybrid_1 = pt.kron(pt.kron(psi_1, Z2_), Z2_)
        psi_hybrid_1 = Controlled_U(t, 0, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(2 * t, 1, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(4 * t, 2, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(8 * t, 3, m) @ psi_hybrid_1
        psi_hybrid_1 = Controlled_U(16 * t, 4, m) @ psi_hybrid_1
        psi_hybrid_1 = pt.kron(QFT_5(0,  1,  2,  3,  4).contiguous() , pt.eye(d**2, dtype=pt.cfloat)) @ psi_hybrid_1
        I2 = pt.kron(0.5 * (I + Z[2]), pt.eye(d ** 2, dtype=pt.cfloat))
        I3 = pt.kron(0.5 * (I + Z[3]), pt.eye(d ** 2, dtype=pt.cfloat))
        I4 = pt.kron(0.5 * (I + Z[4]), pt.eye(d ** 2, dtype=pt.cfloat))
        ipr = pt.vdot(psi_hybrid_1, I0 @ I1 @ I2 @ I3 @ I4 @ psi_hybrid_1)

    return np.real(ipr.item())

m = np.linspace(-1, 0, 20)
IPR_estimate = []
IPR_ED = []
# Exact diagonalization
def IPR_ed(psi, H):
    E, V = pt.linalg.eigh(H)
    ED = 0
    for i in range(len(E)):
        ED += abs(pt.vdot(V[:, i], psi)) ** 4
    return ED

for k in m:

    IPR = IPR_algorithm(k, L_qubit)
    IPR_ED.append(IPR_ed(Z2_, U_t(1, k)[1]))
    IPR_estimate.append(IPR)




# np.savetxt('PXP_8_5_3_26.csv', IPR_estimate, delimiter=',')
plt.plot(m, IPR_estimate, label=f"{L_qubit}Q")
plt.plot(m, IPR_ED, label="ED")

filename = 'PXP_est' + str(L_qubit) + '.csv'
np.savetxt('PXP_ed.csv', IPR_ED, delimiter=',')

np.savetxt(filename, IPR_estimate, delimiter=',')

plt.legend()
plt.xlabel('$m$')
plt.ylabel('IPR')
plt.title('IPR')
plt.show()





