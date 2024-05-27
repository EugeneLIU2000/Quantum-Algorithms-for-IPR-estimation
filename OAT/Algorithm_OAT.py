import numpy as np
 
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.sparse import kron

def get_sparse(A):
    return sparse.coo_matrix(A)

# Now we first define the top qubit system
had = 1.0 / np.sqrt(np.array(2)) * np.array([[1, 1], [1, -1]]) + 1j * 0
had = get_sparse(had)

# refers to the number of sites in OAT ZZ model
N_oat = 4

# the total qubits in the quantum algorithm: 3 * N + 1
L_qubit = 3 * N_oat + 1

# Operators
id_local = np.eye(2, dtype=np.complex64)
sigma_x = np.array([[0, 1], [1., 0j]])
sigma_y = np.array([[0, -1j], [1.j, 0j]])
sigma_z = np.array([[1, 0j], [0., -1]])
p0 = np.array([[1, 0j], [0., 0]])
p1 = np.array([[0, 0j], [0., 1]])


# Transform to sparse matrices
id_local = get_sparse(id_local)
sigma_x = get_sparse(sigma_x)
sigma_y = get_sparse(sigma_y)
sigma_z = get_sparse(sigma_z)
p0 = get_sparse(p0)
p1 = get_sparse(p1)

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
    P0 = {}
    P1 = {}
    Had = {}


    for qubit_i in range(L):  # Loop over indices on a 2-dimensional grid (i_x,i_y)
        X[qubit_i] = get_chain_operator(sigma_x, L, qubit_i)  # Define operator X_i acting on spin (i_x,i_y)
        Y[qubit_i] = get_chain_operator(sigma_y, L, qubit_i)  # Define operator Y_i acting on spin (i_x,i_y)
        Z[qubit_i] = get_chain_operator(sigma_z, L, qubit_i)  # Define operator Z_i acting on spin (i_x,i_y)
        P0[qubit_i] = get_chain_operator(p0, L, qubit_i)  # Define operator Z_i acting on spin (i_x,i_y)
        P1[qubit_i] = get_chain_operator(p1, L, qubit_i)  # Define operator Z_i acting on spin (i_x,i_y)
        Had[qubit_i] = get_chain_operator(had, L, qubit_i)
    return Id, X, Y, Z, P0, P1, Had


# for the circuit
I, X, Y, Z, P0, P1, Had = get_chain_operators(L_qubit)

# for the hamiltonian
id, x, y, z, pp0, pp1, ha = get_chain_operators(N_oat)



def CNOT(i, j):
    return P0[i] @ I + P1[i] @ X[j]


def SWAP(i, j):
    return 0.5 * (X[i] @ X[j] + Z[i] @ Z[j] + Y[i] @ Y[j] + I)



# the controlled-SWAP gate, controlled by qubit, and swap two qudit system
def cSWAP(k, i, j):
    return P0[k] @ I + P1[k] @ SWAP(i, j)


def ipr_algorithm(V):

    psi = np.zeros((2**N_oat, 1), dtype=np.complex64)
    for i in range(2**N_oat):
        psi[i] = V[i]

    qubit = np.zeros((2,1), dtype=np.complex64)
    qubit[0] = 1 # initialization of the first qubit
    qubit = had @ qubit


    # Initialize the qudit register
    qudit_register = np.zeros((2**N_oat,1), dtype=np.complex64)
    qudit_register[0] = 1

    # Initializing state
    Psi = kron(qubit, kron(psi, kron(psi, qudit_register)))

    for i in range(N_oat):
        Psi = CNOT(i+N_oat+1, i+2*N_oat+1) @ Psi # CNOT gates between psi and qudit_register


    for i in range(N_oat):
        Psi = cSWAP(0, i+1, i+N_oat+1) @ Psi # controlled-SWAP gates

    Psi = Had[0] @ Psi

    # The measurement on first qubit
    M_qubit = P0[0]
    # Prob. of first qubit to 0
    prob_0 = Psi.T.conjugate() @ M_qubit @ Psi
    return (np.abs(prob_0.todense()[0, 0]) - 0.5) * 2


def IPR_ED(V):
    psi = V
    return np.sum(np.abs(psi)**4)


result_ED = []
result_algorithm = []


#%%

def get_range(i, j):
    return np.arange(i, j)

T_vec = np.linspace(0, np.pi/2, 50)
for t_idx, t in enumerate(T_vec):
    
    

    # Initialise Hamiltonian
    H = 0
    psi_0 = np.zeros((2 ** N_oat, 1), dtype=np.complex64)
    psi_0[0] = 1
    for i in range(N_oat):
        psi_0 = ha[i] @ psi_0
    # Sum each spin's contribution to the Hamiltonian
    for i in get_range(0,N_oat):
        # Get the index of j and j+1, returning to 0 at j=n+1
        z_dot_z = 0  # Initialise S_j.S_j+1

        z_dot_z += z[i] @ z[(i+1)%N_oat]
        H += z_dot_z

    V = expm(-1j * H.todense() * t) @ psi_0

    for i in range(N_oat):
        V = ha[i] @ V

    # E, V = eigsh(H, k=1, which='SA')
    result_ED.append(IPR_ED(V))
    result_algorithm.append(ipr_algorithm(V))

    print("loop index:", t_idx)


plt.plot(T_vec, result_ED, label="ED, N = %d" %N_oat)
plt.plot(T_vec, result_algorithm,  marker=".",  markersize=6, linestyle='None', label="Estimation, N = %d" %N_oat)


plt.legend()
plt.xlabel('t')
plt.ylabel('IPR')
plt.title('IPR of OAT')
plt.show()

# Collecting the data
np.savetxt('IPR_ED.csv', result_ED, delimiter=',')
np.savetxt('IPR_algorithm.csv', result_algorithm, delimiter=',')


