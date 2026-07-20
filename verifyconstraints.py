import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
import os
if not os.path.exists('figs'):
    os.mkdir('figs')

# Parameters for simulation
N = 7  # Fock space truncation per mode (dim = N^3 = 216, manageable)
num_modes = 3
times = np.linspace(0, 100, 200)  # Evolution times
g = 1.0  # Coupling strength for each H_u term

# Constraint: 3*n1 + n2 + n3 = 6 (simplified knapsack for demo; all n_i >=0 integers)
constraint_coeffs = [3, 1, 1]
# target_value = 5

# Integer null space basis vectors (small steps for connectivity)
# u1 = [1, -2, 0]: 2*1 + (-2) + 0 = 0
# u2 = [0, 1, -1]: 0 + 1 + (-1) = 0
null_space_basis = [
    np.array([1, -3, 0]),
    np.array([0, 1, -1])
]


# Define annihilation and creation operators for each mode
a = [None] * num_modes
ad = [None] * num_modes
for i in range(num_modes):
    ops = [qt.qeye(N)] * num_modes
    ops[i] = qt.destroy(N)
    a[i] = qt.tensor(ops)
    ad[i] = a[i].dag()

# Construct the driver Hamiltonian H_d = sum_u g * (O_u + O_u^\dagger)
H_d = 0 * a[0]  # Zero operator to start
for u in null_space_basis:
    # Build O_u = product over i: (ad_i)^{u_i} if u_i > 0 else a_i^{|u_i|} if u_i < 0 else id
    O_u = 1  # Identity
    for i in range(num_modes):
        if u[i] > 0:
            O_u = O_u * (ad[i] ** u[i])
        elif u[i] < 0:
            O_u = O_u * (a[i] ** abs(u[i]))
        # else: identity (no op)
    H_u = g * (O_u + O_u.dag())
    H_d += H_u

print("Driver Hamiltonian H_d constructed.")
print(f"Null space basis: {null_space_basis}")

# Hardcoded initial state label
initial_fock = [0,0,6]
initial_fock2 = [0,6,0]
target_value = sum(constraint_coeffs[j] * initial_fock[j] for j in range(3))
print(f"H_d norm: {H_d.norm():.2f}")
print(f"target_value: {target_value}")
initial_state = qt.tensor(*[qt.fock(N, f) for f in initial_fock])
initial_state2 = qt.tensor(*[qt.fock(N, f) for f in initial_fock2])
surporse_initial = (initial_state + initial_state2).unit()
# Evolve the state under H_d (pure state, no dissipation)
result = qt.mesolve(H_d, surporse_initial, times, [], [])

# Function to compute Fock probabilities at time t_idx
def get_fock_probs(rho):
    probs = np.zeros((N, N, N))
    for n1, n2, n3 in product(range(N), repeat=3):
        basis_state = qt.tensor(qt.fock(N, n1), qt.fock(N, n2), qt.fock(N, n3))
        probs[n1, n2, n3] = abs(basis_state.overlap(rho))**2
    return probs

# Function to compute constraint satisfaction rate at time t_idx
def get_satisfaction_rate(probs):
    valid_probs = 0.0
    for n1, n2, n3 in product(range(N), repeat=3):
        if sum(constraint_coeffs[j] * [n1, n2, n3][j] for j in range(3)) == target_value:
            valid_probs += probs[n1, n2, n3]
    return valid_probs

# Compute probs and satisfaction over time
probs_over_time = np.zeros((len(times), N, N, N))
satisfaction_rates = np.zeros(len(times))
for t_idx, t in enumerate(times):
    rho_t = result.states[t_idx]
    probs_t = get_fock_probs(rho_t)
    probs_over_time[t_idx] = probs_t
    satisfaction_rates[t_idx] = get_satisfaction_rate(probs_t)
#  3*n1 + n2 + n3 = 6 
# Select some interesting states to plot (valid ones connected by the driver)
plot_states = [
    (1, 1, 2),  # Initial
    (1, 2, 1),  # Via u2
    (1, 0, 3),  # Via u2^\dagger
    (1, 3, 0),   # Another valid state
    (2, 0, 0),  # Another valid state
    (0, 4, 2),  # Via u1^\dagger
    (0, 2, 4),  # Via u1^\dagger
    (0, 1, 5),  # Further via u1^\dagger from another, etc.
    (0, 5, 1),
    (0, 6, 0), # Boundary
    (0, 0, 6)  # Boundary
]
state_labels = [f"|{n1},{n2},{n3}>" for n1, n2, n3 in plot_states]

colors = plt.cm.viridis(np.linspace(0, 1, len(plot_states)))
# Extract probabilities for these states over time
state_probs = np.zeros((len(times), len(plot_states)))
for s_idx, (n1, n2, n3) in enumerate(plot_states):
    state_probs[:, s_idx] = probs_over_time[:, n1, n2, n3]

# Plot 1: Probabilities of selected Fock states over time
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for i in range(len(plot_states)):
    plt.plot(times, state_probs[:, i], label=state_labels[i], linewidth=2, color=colors[i])
plt.xlabel('Evolution Time $t$')
plt.ylabel('Probability $|\\langle n | \\psi(t) \\rangle|^2$')
plt.title('Fock State Probabilities under $H_d$ Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Constraint satisfaction rate over time
plt.subplot(1, 2, 2)
plt.plot(times, satisfaction_rates, 'r-', linewidth=3, label='Satisfaction Rate')
## plot the mean value for each state
for i in range(len(plot_states)):
    max_val = np.max(state_probs[10:, i])
    plt.hlines(max_val, times[0], times[-1], colors= colors[i], linestyles='dashed', alpha=0.5)
    plt.text(times[-1], max_val, f"{state_labels[i]} max={max_val:.2f}", fontsize=8, verticalalignment='bottom', horizontalalignment='right')
plt.xlabel('Evolution Time $t$')
plt.ylabel('Satisfaction Rate')
plt.title('Constraint Satisfaction Rate (should stay at 1.0)')
plt.ylim(0, 1.01)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
## add a title to the entire figure
plt.suptitle('Initial state |x1,x2,x3> = {}'.format(initial_fock)+
             '\n , constraint: {}'.format(" + ".join(f"{c}*x{i+1}" for i,c in enumerate(constraint_coeffs)) 
                                          + f" = {target_value}"), y=-0.02)
plt.savefig('figs/verification_plot.svg')
plt.show()  # Display if in interactive mode

# Print final satisfaction and top states
print(f"\nFinal satisfaction rate: {satisfaction_rates[-1]:.6f} (should be 1.0)")
print("Top 5 final Fock states by probability:")
final_probs = probs_over_time[-1]
flat_probs = [( (n1,n2,n3), prob ) for n1,n2,n3 in product(range(N), repeat=3) for prob in [final_probs[n1,n2,n3]] if prob > 1e-3]
flat_probs.sort(key=lambda x: x[1], reverse=True)
for i, ((n1,n2,n3), prob) in enumerate(flat_probs[:5]):
    constr_val = sum(constraint_coeffs[j] * [n1,n2,n3][j] for j in range(3))
    valid = "VALID" if constr_val == target_value else "INVALID"
    print(f"  |{n1},{n2},{n3}>: {prob:.4f} ({valid}, constr={constr_val})")