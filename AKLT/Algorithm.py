# from Operator import *
# from Hamiltonian import *


import numpy as np

import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.sparse import kron
from scipy.sparse import eye
from scipy.sparse.linalg import eigsh
import scipy



import numpy as np

import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.sparse import kron

def get_sparse(A):
    return sparse.coo_matrix(A)

# Now we first define the top qubit system
had = 1.0 / np.sqrt(np.array(2)) * np.array([[1, 1], [1, -1]]) + 1j * 0
c0 = np.array([[1., 0+ 0j], [0, 0]])
c1 = np.array([[0., 0+ 0j], [0, 1]])

had = get_sparse(had)
c0 = get_sparse(c0)
c1 = get_sparse(c1)




# the number of sites in AKLT model
N_aklt = 5

# the total qudits in the quantum algorithm
L_qubit = N_aklt*3

# Operators
id_local = np.eye(3, dtype=np.complex64)
sigma_x = np.sqrt(1/2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1., 0j]])
sigma_y = -1j * np.sqrt(1/2) * np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])
sigma_z = np.array([[1., 0, 0], [0, 0, 0], [0, 0.j, -1]])

shift1 = np.array([[0, 0, 1], [1, 0, 0j], [0, 1., 0]])
shift2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0., 0j]])

P0 = np.array([[1, 0., 0], [0, 0, 0], [0, 0, 0j]])
P1 = np.array([[0, 0., 0], [0, 1, 0], [0, 0, 0j]])
P2 = np.array([[0, 0., 0], [0, 0, 0], [0, 0j, 1]])

K3 = np.array([[1, 0., 0], [0, 0, 1], [0, 1, 0.j]]) # gate for implementing SWAP gate in qutrit case

# Transform to sparse matrices
id_local = get_sparse(id_local)
sigma_x = get_sparse(sigma_x)
sigma_y = get_sparse(sigma_y)
sigma_z = get_sparse(sigma_z)

shift1 = get_sparse(shift1)
shift2 = get_sparse(shift2)

P0 = get_sparse(P0)
P1 = get_sparse(P1)
P2 = get_sparse(P2)
K3 = get_sparse(K3)


#%%



def get_Identity(k):  # returns k-tensor product of the identity operator, ie. Id^k
    Id = id_local
    for i in range(0, k-1):
        Id = kron(Id, id_local)
    return Id


def get_chain_operator(A, L, i):
    if (i == 0):
        Op = kron(A, get_Identity(L - 1))

    elif (i == L-1):
        Op = kron(get_Identity(L - 1), A)

    elif (i > 0 and i < L-1):
        Op = kron(get_Identity(i), kron(A, get_Identity(L - i - 1)))

    return Op


def get_chain_operators(L):
    Id = get_chain_operator(id_local, L, 1)
    X = {}
    Y = {}
    Z = {}
    s1 = {}
    s2 = {}
    p0 = {}
    p1 = {}
    p2 = {}
    k3 = {}

    for qubit_i in range(L):  # Loop over indices on a 2-dimensional grid (i_x,i_y)
        X[qubit_i] = get_chain_operator(sigma_x, L, qubit_i)  # Define operator X_i acting on spin (i_x,i_y)
        Y[qubit_i] = get_chain_operator(sigma_y, L, qubit_i)  # Define operator Y_i acting on spin (i_x,i_y)
        Z[qubit_i] = get_chain_operator(sigma_z, L, qubit_i)  # Define operator Z_i acting on spin (i_x,i_y)
        s1[qubit_i] = get_chain_operator(shift1, L, qubit_i)
        s2[qubit_i] = get_chain_operator(shift2, L, qubit_i)
        p0[qubit_i] = get_chain_operator(P0, L, qubit_i)
        p1[qubit_i] = get_chain_operator(P1, L, qubit_i)
        p2[qubit_i] = get_chain_operator(P2, L, qubit_i)
        k3[qubit_i] = get_chain_operator(K3, L, qubit_i)

    return Id, X, Y, Z, s1, s2, p0, p1, p2, k3


# for the circuit
I, X, Y, Z, s1, s2, p0, p1, p2, k3 = get_chain_operators(L_qubit)

# for the hamiltonian
id, x, y, z, ss1, ss2, pp0, pp1, pp2, kk3 = get_chain_operators(N_aklt)



def SUM(i, j):
    return p0[i] @ I + p1[i] @ s1[j] + p2[i] @ s2[j]


def SWAP(i, j):
    return SUM(i, j) @ k3[i] @ SUM(j, i) @ k3[i] @ SUM(i, j) @ k3[j]



# the controlled-SWAP gate, controlled by qubit, and swap two qudit system
def cSWAP(i, j):
    return kron(c0, I) + kron(c1, SWAP(i, j))


def ipr_algorithm(V):

    psi = np.zeros((3**N_aklt, 1), dtype=np.cfloat)
    for i in range(3**N_aklt):
        psi[i] = V[i, 0]

    qubit = np.zeros((2,1), dtype=np.cfloat)
    qubit[0] = 1 # initialization of the first qubit
    qubit = had @ qubit

    # Initialize the qudit register
    qudit_register = np.zeros((3 ** N_aklt,1), dtype=np.cfloat)
    qudit_register[0] = 1

    # Intermediate state
    Psi_0 = kron(psi, kron(psi, qudit_register))

    for i in range(N_aklt):
        Psi_0 = SUM(i+N_aklt, i+2*N_aklt) @ Psi_0  # generalized CNOT gates between psi and qudit_register

    # Now we link the first qubit to qudits
    Psi_1 = kron(qubit, Psi_0)


    for i in range(N_aklt):
        Psi_1 = cSWAP(i, i+N_aklt) @ Psi_1 # controlled-SWAP gates

    Psi_1 = kron(had, I) @ Psi_1

    # The measurement on first qubit
    M_qubit = kron(c0, I)
    # Prob. of first qubit to 0
    prob_0 = Psi_1.T.conjugate() @ M_qubit @ Psi_1
    return (np.abs(prob_0.todense()[0, 0]) - 0.5) * 2


def IPR_ED(V):
    psi = V[:, 0]
    return np.sum(np.abs(psi)**4)


result_ED = []
result_algorithm = []


#%%

def get_range(i, j):
    return np.arange(i, j)

B_vec = np.linspace(0, 10, 100)

J1 = 1./2
J2 = 1./6.
for B_idx, B in enumerate(B_vec):
    # Initialise Hamiltonian
    H = 0

    # Sum each spin's contribution to the Hamiltonian
    for i in get_range(0,N_aklt-1):

        s_dot_s = 0  # Initialise S_j.S_j+1

        s_dot_s += x[i] @ x[i+1]
        s_dot_s += y[i] @ y[i+1]
        s_dot_s += z[i] @ z[i+1]
        H += J1 * s_dot_s + J2 * s_dot_s @ s_dot_s + 1/3 * id + B/N_aklt * z[i]

    H += B/N_aklt * z[N_aklt-1]

    E, V = eigsh(H, k=1, which='SA')
    result_ED.append([B, IPR_ED(V)])
    result_algorithm.append([B, ipr_algorithm(V)])

    print("loop index:", B_idx)
    
result_ED = np.array(result_ED)
result_algorithm = np.array(result_algorithm)

#%%

plt.plot(result_ED[:,0], result_ED[:,1], ls='-', color = 'black', label="ED, N = %d" % N_aklt)
plt.plot(result_algorithm[:,0], result_algorithm[:,1],  ls = '--', color = 'red',  label="Estimation, N = %d" %N_aklt)


plt.legend()
plt.xlabel('B')
plt.ylabel('IPR')
plt.title('IPR of AKLT')
plt.show()

# Collecting the data
parameters = 'N_AKLT_sites.' + str(N_aklt)
np.savetxt('IPR_AKLT_ED_' + parameters + '.dat', result_ED )
np.savetxt('IPR_AKLT_algorithm_' + parameters + '.dat', result_algorithm)


