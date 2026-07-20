import numpy as np
from qutip import (Qobj, about, basis, destroy, enr_destroy, enr_fock,
                   enr_state_dictionaries, identity, liouvillian, mesolve,
                   plot_expectation_values, tensor)
from qutip.core.energy_restricted import EnrSpace

N = 4  # number of systems
M = 2  # number of cavity states
dims = [M, 2] * N  # dimensions of JC spin chain
excite = 1  # total number of excitations
init_excite = 1  # initial number of excitations
# Setup to Calculate Time Evolution
def solve(d, psi0):
    # annihilation operators for cavity modes
    a = d[::2]
    # atomic annihilation operators
    sm = d[1::2]

    # notice the ordering of annihilation and creation operators
    H0 = sum([aa.dag() * aa for aa in a]) + sum([s.dag() * s for s in sm])

    # atom-cavity couplings
    Hint_ac = 0
    for n in range(N):
        Hint_ac += 0.5 * (a[n].dag() * sm[n] + sm[n].dag() * a[n])

    # cavity-cavity couplings
    Hint_cc = 0
    for n in range(N - 1):
        Hint_cc += 0.9 * (a[n].dag() * a[n + 1] + a[n + 1].dag() * a[n])

    H = H0 + Hint_ac + Hint_cc

    e_ops = [x.dag() * x for x in d]
    c_ops = [0.01 * x for x in a]

    times = np.linspace(0, 250, 1000)
    L = liouvillian(H, c_ops)
    opt = {"nsteps": 5000, "store_states": True}
    result = mesolve(H, psi0, times, c_ops, e_ops, options=opt)
    return result, H, L
# Regular QuTiP States and Operators
d = [
    tensor(
        [
            destroy(dim1) if idx1 == idx2 else identity(dim1)
            for idx1, dim1 in enumerate(dims)
        ]
    )
    for idx2, _ in enumerate(dims)
]
psi0 = tensor(
    [
        basis(dim, init_excite) if idx == 1 else basis(dim, 0)
        for idx, dim in enumerate(dims)
    ]
)
# Regular operators of different systems commute as they belong to different Hilbert spaces. Example:

d[0].dag() * d[1] == d[1] * d[0].dag()
# True
# Solving the time evolution:

res1, H1, L1 = solve(d, psi0)
print(f"Run time: {res1.stats['run time']}s")
# Using ENR States and Operators
d_enr = enr_destroy(dims, excite)
init_enr = [init_excite if n == 1 else 0 for n in range(2 * N)]
psi0_enr = enr_fock(dims, excite, init_enr)
# Using ENR states forces us to give up on the standard tensor structure of multiple Hilbert spaces. Operators for different systems therefore generally no longer commute:

d_enr[0].dag() * d_enr[1] == d_enr[1] * d_enr[0].dag()

res2, H2, L2 = solve(d_enr, psi0_enr)
print(f"Run time: {res2.stats['run time']}s")

fig, axes = plot_expectation_values([res1, res2])
fig.set_figwidth(10)
fig.set_figheight(8)
for idx, ax in enumerate(axes):
    if idx % 2:
        ax.set_ylabel(f"Atom {idx//2}")
    else:
        ax.set_ylabel(f"Cavity {idx//2}")
    ax.set_ylim(-0.1, 1.1)
    ax.grid()
fig.tight_layout()
