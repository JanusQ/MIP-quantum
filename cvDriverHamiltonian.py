import qutip as qt
import numpy as np

N = 30  # Fock truncation per mode
a = qt.destroy(N)
ad = a.dag()
num_modes = 3
a1 = qt.tensor(a, qt.qeye(N))
ad1 = a1.dag()
a2 = qt.tensor(qt.qeye(N), a)
ad2 = a2.dag()
Xs = []
Ps = []
for i in range(num_modes):
    Xs.append((a + a.dag()) / np.sqrt(2))
    Ps.append(-1j * (a - a.dag()) / np.sqrt(2))
    Xs[-1] = qt.tensor([qt.qeye(N)] * i + [Xs[-1]] + [qt.qeye(N)] * (num_modes - i - 1))
    Ps[-1] = qt.tensor([qt.qeye(N)] * i + [Ps[-1]] + [qt.qeye(N)] * (num_modes - i - 1))
    
print((a * ad - ad * a  -   qt.qeye(N)).norm())

C = Xs[0] + Xs[1] + Xs[2]

gs = np.array([1.0, 1.0, 1.0])
u =  [[1, -1, 0],
      [0, -1, 1],]
Hd = 0 * Ps[0] 
for idx, ui in enumerate(u):
    for i in range(num_modes):
        Hd += ui[i] * gs[idx] * Ps[i]

# ## check [Hd, C] = 0
# print("Commutator [Hd, C]:")
# print((Hd * C - C * Hd).norm())  # should be close to 0
from positioneigenstate import approximate_position_eigenstate
psi0 = qt.tensor(approximate_position_eigenstate(N, 3.0), qt.basis(N, 0), qt.basis(N, 0))

times = np.linspace(0,40, 80)

result = qt.sesolve(Hd, psi0, times, e_ops=[C,Xs[0], Xs[1], Xs[2]])
import matplotlib.pyplot as plt
plt.plot(times, result.expect[0], 'o-', label="<C>")
for i in range(num_modes):
    plt.plot(times, result.expect[1+i], 'o-', label=f"<X_{i+1}>")
plt.xlabel("time")
plt.ylabel("<C>")
plt.legend()
plt.savefig("verify_constraints.png", dpi=300)
# # Expectations (constant across t)
# print("Expectation values of C at different times:")
# print(result.expect[0])
# for i in range(num_modes):
#     print(f"Expectation values of X_{i+1} at different times:")
#     print(result.expect[1 + i])
