import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import numpy as np
from qutip import *

def optimal_r(N, x0, lambda_x=1/np.sqrt(2), r_vals=None):
    if r_vals is None:
        r_vals = np.linspace(0, 3, 61)   # search grid

    a = destroy(N)
    x_op = (a + a.dag()) * lambda_x
    alpha = x0 / (2 * lambda_x)

    errors = []
    for r in r_vals:
        D = displace(N, alpha)
        S = squeeze(N, r)
        psi = D * S * basis(N, 0)
        mean_x = expect(x_op, psi)
        errors.append(abs(mean_x - x0))

    r_opt = r_vals[np.argmin(errors)]
    return r_opt, errors

def approximate_position_eigenstate(N, x0, plot=False):
    """
    # Example usage:
    N = 10  # Fock truncation per mode
    x0 = 2.0  # Target position eigenvalue
    psi = approximate_position_eigenstate(N, x0)
    """
    N = 10
    x0 = 2.0
    lambda_x = 1/np.sqrt(2)
    r_opt, errs = optimal_r(N, x0)
    print("Optimal r ≈", r_opt)

    # ------------------------
    # Operators
    # ------------------------
    a = destroy(N)

    # Displacement parameter: shift mean position to x0
    alpha = x0 / (2 * lambda_x)

    # Gates
    D = displace(N, alpha)      # displacement operator
    S = squeeze(N, r_opt)           # squeezing operator

    # Prepare state: squeezed + displaced vacuum
    psi = D * S * basis(N, 0)
    # ------------------------
    # Check expectation value of position
    # ------------------------
    x_op = (a + a.dag()) * lambda_x
    print("⟨x⟩ =", expect(x_op, psi))
    print("Δx =", np.sqrt(expect(x_op**2, psi) - expect(x_op, psi)**2))
    
    # ------------------------
    # Plot Wigner function
    # ------------------------
    if plot:
        xvec = np.linspace(-10, 10, 200)
        W = wigner(psi, xvec, xvec)

        plt.contourf(xvec, xvec, W, 100, cmap='RdBu_r')
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("p")
        plt.title("Approximate position eigenstate at x0 = {:.2f}".format(x0))
        plt.savefig("approx_position_eigenstate.png", dpi=300)

    return psi
