import numpy as np
import scipy.sparse
import scipy.sparse.linalg

xQB = np.array([[0, 1], [1, 0]])
yQB = np.array([[0, -1j], [1j, 0]])

"""Matrix representation """

def R_matrix(theta, phi):
    """Return the matrix of R^phi(theta) gate."""
    H = np.cos(phi) * xQB + np.sin(phi) * yQB
    arg = -1j * (theta / 2) * H
    return scipy.sparse.linalg.expm(scipy.sparse.csc_matrix(arg)).toarray()

def Rmode_matrix(theta, cutoff):
    """Qumode number operator rotation: exp(-i * theta * n̂)
    
    Args:
        theta (real): rotation angle
        cutoff (int): qumode Hilbert space cutoff
    
    Returns:
        csc_matrix: cutoff x cutoff sparse diagonal matrix
    """
    # 对角元素: exp(-i * theta * n) for n = 0..cutoff-1
    diag = np.exp(-1j * theta * np.arange(cutoff))
    return scipy.sparse.diags(diag, format="csc").toarray()


    
    
    
def SQR_matrix(thetalist, philist, cutoff):
    """
    Build the matrix for SQR gate.
    thetalist, philist: lists of same length
    cutoff: qumode Hilbert space dimension
    """
    if len(thetalist) != len(philist):
        raise ValueError("thetalist and philist must have the same length")

    dim_qubit = 2
    total_dim = dim_qubit * cutoff
    SQR_mat = scipy.sparse.csc_matrix((total_dim, total_dim), dtype=complex)

    for n, (theta, phi) in enumerate(zip(thetalist, philist)):
        # R gate for qubit
        Rn = R_matrix(theta, phi)
        # projector |n><n| in qumode space
        ket_n = np.zeros(cutoff)
        ket_n[n] = 1
        proj_n = scipy.sparse.csc_matrix(np.outer(ket_n, ket_n))
        # Kronecker product: Rn ⊗ |n><n|
        term = scipy.sparse.kron(Rn, proj_n, format="csc")
        SQR_mat += term

    return SQR_mat.toarray()

def CRi_matrix(theta, cutoff):
    """Controlled qubit–qumode number rotation:
        exp(-i * theta/2 * Z ⊗ n̂)

    Args:
        theta (real): rotation angle
        cutoff (int): qumode Hilbert space cutoff

    Returns:
        csc_matrix: (2*cutoff) x (2*cutoff) sparse matrix
    """
    # qumode rotations for |0> and |1>
    R_plus  = Rmode_matrix(+theta/2, cutoff)  # for Z=+1
    R_minus = Rmode_matrix(-theta/2, cutoff)  # for Z=-1

    # 2x2 block diagonal: [R_plus, R_minus]
    return scipy.sparse.block_diag([R_plus, R_minus], format="csc").toarray()


"""Gates """


def R(circuit, qubit,theta,phi):
    """
    R gate is a single-qubit gate that is a rotation around the XY-axis.
    \[
    R^\phi(\theta)=exp(-i\theta/2 (X cos(\phi) +  Y sin(\phi)))
    \]
    """
    circuit.unitary(R_matrix(theta, phi), [qubit])

def SQR(circuit, qubit, qumode, thetalist,philist,cutoff):
    """
    SQR gate is a Transom-Mode gate. which is
    
    \[
    SQR_{i,j}(\vec{\theta},\vec{\phi})=\sum_{n} R^{\phi_n}_{i}(\theta_n) \otimes |n\rangle\langle n|_j
    \]
    where i is a qubit and j is a qumode
    """
    U = SQR_matrix(thetalist, philist, cutoff)
    circuit.unitary(U, [qubit]+qumode)


def cv_R(circuit, qumode, theta, cutoff):
    circuit.unitary(Rmode_matrix(theta, cutoff), qumode)

def cv_CRi(circuit, qubit, qumode, theta, cutoff):
    circuit.unitary(CRi_matrix(theta, cutoff), [qubit]+qumode)
        
def cv_CRi_dag(circuit, qubit, qumode, theta, cutoff):
    circuit.unitary(CRi_matrix(theta, cutoff).conj().T, [qubit]+qumode)

def add_RR(circ, anc, i, j, theta, cutoff):
    """
    Add RR_{i,j}(theta) to circuit using ancilla anc, qumodes i and j.
    """
    import numpy as np
    
    def get_pi_vec_k(k, cutoff):
        return [np.pi if (n // (2**k)) % 2 == 1 else 0 for n in range(cutoff)]
    
    K = int(np.ceil(np.log2(cutoff)))
    
    for k in range(K):
        pi_vec_k = get_pi_vec_k(k, cutoff)
        phi = (2**k) * theta / 2
        
        # SQR(π⃗_k, 0⃗) on ancilla-qumode (anc, i)
        SQR(circ, anc, i, pi_vec_k, [0]*cutoff, cutoff)
        
        # R_j(phi)
        cv_R(circ, j, phi, cutoff)  # Or circ.R(j, phi) depending on your API
        
        # CR_{i,j}†(phi)
        cv_CRi_dag(circ, anc, j, phi, cutoff)
        
        # SQR(-π⃗_k, 0⃗)
        SQR(circ, anc, i, [-x for x in pi_vec_k], [0]*cutoff,cutoff)
