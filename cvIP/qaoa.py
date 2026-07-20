import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
import sympy as sp
from typing import List, Tuple, Optional, Union, Dict


class BosonicQAOAIPSolver:
    """
    Bosonic QAOA solver for integer programs with linear or quadratic constraints:
    max c^T x s.t. constraint, x >= 0 integer.
    
    Encoding: x_i <-> n_i (photon number in mode i).
    Constraint subspace S_c: { |n> | constraint holds }.
    Driver H_M: sum_u (O_u + O_u^dagger) over integer nullspace basis of A.
    Cost H_C = - sum c_i n_i.
    
    Args:
        A: Constraint matrix (m x d integer for linear; d x d symmetric integer for quadratic).
        b: RHS (m-vector integer for linear; scalar [b] for quadratic).
        c: Objective coeffs (d, maximize c^T x).
        N: Fock truncation per mode.
        p: QAOA layers.
        num_modes: d (inferred from A/c if not given).
        g: Driver coupling strength.
        maxiter: Optimization iterations.
        seed: Random seed.
        constraint_type: 'linear' or 'quadratic'.
        circuit_type: "beta_gamma" or "multi_beta".
    """
    
    def __init__(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        N: int = 7,
        p: int = 2,
        g: float = 1.0,
        maxiter: int = 150,
        seed: int = 42,
        num_modes: Optional[int] = None,
        constraint_type: str = "linear",
        circuit_type: str = "beta_gamma"
    ):
        self.constraint_type = constraint_type
        self.A = np.array(A, dtype=int)
        self.b = np.array(b, dtype=int)
        self.c = np.array(c, dtype=float)
        self.N = N
        self.p = p
        self.g = g
        self.maxiter = maxiter
        self.seed = seed
        self.num_modes = num_modes or self.c.shape[0] or self.A.shape[1]
        
        # Validate dimensions
        if constraint_type == "linear":
            if self.A.shape[1] != self.num_modes or self.c.shape[0] != self.num_modes:
                raise ValueError("Dimensions mismatch: A (m x d), b (m), c (d).")
            if self.A.shape[0] != self.b.shape[0]:
                raise ValueError("A rows != b length.")
        else:  # quadratic
            if self.A.shape[0] != self.A.shape[1] or self.A.shape[0] != self.num_modes:
                raise ValueError("For quadratic, A must be d x d.")
            if self.b.shape[0] != 1:
                raise ValueError("For quadratic, b must be scalar (array of len 1).")
            if not np.allclose(self.A, self.A.T):
                raise ValueError("A must be symmetric for quadratic.")
        
        np.random.seed(seed)
        
        # Compute integer nullspace basis using sympy
        self.null_space_basis = self._compute_integer_nullspace()
        print(f"Computed integer nullspace basis with {len(self.null_space_basis)} vectors.")
        for null_vec in self.null_space_basis:
            print(f"  Null vector: {null_vec}")
        # Build Hamiltonians and operators
        self.a_ops, self.ad_ops, self.n_ops = self._build_operators()
        self.H_C = self._build_cost_hamiltonian()
        self.H_ds = self._build_seperate_driver_hamiltonian()
        self.H_M = sum(self.H_ds)
        self.target_operator = -self.H_C  # For expectation: max c^T n
        self.circuit_type = circuit_type
        if circuit_type == "beta_gamma":
            self.qaoa_circuit = self.qaoa_circuit_beta_gamma
            self.paramlength = 2  # gamma, beta per layer
        elif circuit_type == "multi_beta":
            self.qaoa_circuit = self.qaoa_multi_beta_layer_circuit
            self.paramlength = len(self.null_space_basis)  # beta_i per layer
        elif circuit_type == "multi_beta_oneH":
            self.qaoa_circuit = self.qaoa_multi_beta_oneH_circuit
            self.paramlength = len(self.null_space_basis)  # beta_i per layer
        else:
            raise ValueError("circuit_type must be 'beta_gamma' or 'multi_beta'")
        # Find feasible states for initial and tracking
        self.feasible_states = self._find_feasible_states()
        if not self.feasible_states:
            raise ValueError("No feasible states found in truncation N.")
        
        # Tracked states: sorted by objective descending (top 6 or all if fewer)
        self.initial_state = min(self.feasible_states, key=lambda ns: sum(self.c * ns))
        self.tracked_states = sorted(self.feasible_states, key=lambda ns: sum(self.c * ns), reverse=True)[:6] + [self.initial_state]
        self.tracked_labels = [f"|{','.join(map(str, ns))}⟩ (obj={sum(self.c * ns):.1f})" for ns in self.tracked_states]
        
        print(f"Initialized BosonicQAOAIPSolver: {self.num_modes} modes, {len(self.null_space_basis)} null vectors.")
        print(f"Constraints: {self.constraint_type}, A shape {self.A.shape}. Objective: max {self.c} · x.")
        print(f"Feasible subspace dim: {len(self.feasible_states)} (in truncation N={N}).")
        print(f"initial_state: |{','.join(map(str, self.initial_state))}⟩")
    
    def _compute_integer_nullspace(self) -> List[np.ndarray]:
        """Compute primitive integer basis for ker(A) using sympy nullspace."""
        A_sym = sp.Matrix(self.A)
        ns_rational = A_sym.nullspace()
        if not ns_rational:
            raise ValueError("Nullspace empty; constraints overconstrained.")
        
        # Collect denominators
        denoms = []
        for vec in ns_rational:
            for entry in vec:
                if hasattr(entry, 'is_Rational') and entry.is_Rational:
                    denoms.append(entry.q)  # denominator
        lcm_den = sp.lcm(denoms) if denoms else sp.Integer(1)
        
        # Integer vectors
        int_vecs = []
        for vec in ns_rational:
            int_vec = (lcm_den * vec).applyfunc(lambda x: int(x))
            int_vec = np.array(int_vec).flatten().astype(int)
            gcd = np.gcd.reduce(int_vec)
            if gcd != 0:
                int_vec = int_vec // gcd
            # Flip sign if first non-zero is negative
            if len(int_vec) > 0 and int_vec[int(np.nonzero(int_vec)[0][0])] < 0:
                int_vec = -int_vec
            int_vecs.append(int_vec)
        
        # Remove duplicates
        unique_vecs = [np.array(v) for v in set(tuple(vec) for vec in int_vecs)]
        
        # Optional: Add circle loop vector if >1 basis
        if len(unique_vecs) > 1:
            first_vec = unique_vecs[0]
            last_vec = unique_vecs[-1]
            new_vec = first_vec + last_vec
            new_vec_minus = first_vec - last_vec
            if len(np.nonzero(new_vec)[0]) < len(np.nonzero(new_vec_minus)[0]):
                new_vec = new_vec
            else:
                new_vec = new_vec_minus
            gcd = np.gcd.reduce(new_vec)
            if gcd != 0:
                new_vec = new_vec // gcd
            if len(new_vec) > 0 and new_vec[int(np.nonzero(new_vec)[0][0])] < 0:
                new_vec = -new_vec
            unique_vecs.append(new_vec)
        
        return unique_vecs
    
    def _build_operators(self) -> Tuple[List[qt.Qobj], List[qt.Qobj], List[qt.Qobj]]:
        """Build a_i, a_i^dagger, n_i for all modes."""
        a_ops = []
        ad_ops = []
        n_ops = []
        for i in range(self.num_modes):
            ops_list = [qt.qeye(self.N)] * self.num_modes
            ops_list[i] = qt.destroy(self.N)
            a = qt.tensor(ops_list)
            ad = a.dag()
            n = ad * a
            a_ops.append(a)
            ad_ops.append(ad)
            n_ops.append(n)
        return a_ops, ad_ops, n_ops
    
    def _build_cost_hamiltonian(self) -> qt.Qobj:
        """H_C = - sum c_i n_i."""
        H_C = sum(-self.c[i] * self.n_ops[i] for i in range(self.num_modes))
        return H_C
    
    def _build_seperate_driver_hamiltonian(self) -> List[qt.Qobj]:
        """H_M = g sum_u (O_u + O_u^dagger)."""
        Hds = []
        latex_labels = []
        idx = 1
        for u in self.null_space_basis:
            latex_label = f"$g_{{{idx}}} \\left("
            for i in range(self.num_modes):
                if u[i] > 0:
                    latex_label += f"\\hat{{a}}_{{{i}}}^{{\\dagger {u[i]}}} " if u[i] > 1 else f"\\hat{{a}}_{{{i}}}^\\dagger "
                elif u[i] < 0:
                    latex_label += f"\\hat{{a}}_{{{i}}}^{{{abs(u[i])}}} " if abs(u[i]) > 1 else f"\\hat{{a}}_{{{i}}} "
            latex_label += "+ \\mathrm{h.c.} \\right)$"
            idx += 1
            latex_labels.append(latex_label)
            O_u = 1
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** int(u[i]))
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M = self.g * (O_u + O_u.dag())
            Hds.append(H_M)
        self.latex_label_H_M = " + ".join(latex_labels)
        return Hds
    
    def _find_feasible_states(self) -> List[Tuple[int, ...]]:
        """Enumerate feasible n in [0,N)^d satisfying the constraint."""
        feasible = []
        for ns_tuple in product(range(self.N), repeat=self.num_modes):
            ns = np.array(ns_tuple)
            if self.constraint_type == "linear":
                if np.allclose(self.A @ ns, self.b):
                    feasible.append(tuple(ns))
            else:  # quadratic
                if np.allclose(np.dot(ns, self.A @ ns), self.b[0]):
                    feasible.append(tuple(ns))
        return feasible
    
    def create_initial_state(self, superposition: bool = False) -> qt.Qobj:
        """Superposition of argmax/argmin objective feasible states, or max if not superpose."""
        if not superposition:
            ## select the state with maximum objective
            
            return qt.tensor(*[qt.fock(self.N, n) for n in self.initial_state])
        
        max_ns = max(self.feasible_states, key=lambda ns: sum(self.c * ns))
        min_ns = min(self.feasible_states, key=lambda ns: sum(self.c * ns))
        state1 = qt.tensor(*[qt.fock(self.N, n) for n in max_ns])
        state2 = qt.tensor(*[qt.fock(self.N, n) for n in min_ns])
        return (state1 + state2).unit()
    
    def qaoa_circuit_beta_gamma(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_M(beta) U_C(gamma))."""
        state = initial_state.copy()
        for layer in range(self.p):
            gamma = params[2 * layer]
            beta = params[2 * layer + 1]
            state = (-1j * self.H_C * gamma).expm() * state
            state = (-1j * self.H_M * beta).expm() * state
        return state
    def qaoa_multi_beta_layer_circuit(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_d1(beta1) U_d2(beta2) ...)."""
        betalength = len(self.null_space_basis)
        if len(params) != self.p * (betalength):
            raise ValueError(f"Expected {self.p * (betalength)} params, got {len(params)}.")
            
        state = initial_state.copy()
        for layer in range(self.p):
            for i in range(betalength):
                beta = params[layer * (betalength) + i]
                state = (-1j * self.H_ds[i] * beta).expm() * state
        return state
    def qaoa_multi_beta_oneH_circuit(self, params: np.ndarray, initial_state: qt.Qobj) -> qt.Qobj:
        """p-layer QAOA: prod (U_d(beta1,beta2)...)."""
        betalength = len(self.null_space_basis)
        if len(params) != self.p * (betalength):
            raise ValueError(f"Expected {self.p * (betalength)} params, got {len(params)}.")
            
        state = initial_state.copy()
        for layer in range(self.p):
            betas = params[layer * (betalength):(layer + 1) * (betalength)]
            H_d_layer = sum(betas[i] * self.H_ds[i] for i in range(betalength))
            state = (-1j * H_d_layer).expm() * state
        return state
    
    
    def optimize(self, initial_state: Optional[qt.Qobj] = None) -> dict:
        """COBYLA optimization with history tracking."""
        if initial_state is None:
            initial_state = self.create_initial_state()
        
        # History dict
        self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
        
        def cost_with_history(params: np.ndarray) -> float:
            iter_idx = len(self.iter_history["iter"])
            self.iter_history["iter"].append(iter_idx + 1)
            
            final_state = self.qaoa_circuit(params, initial_state)
            cost_val = qt.expect(self.H_C, final_state)
            self.iter_history["cost"].append(cost_val)
            
            # Track probs for selected states
            current_probs = []
            for ns in self.tracked_states:
                basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
                prob = abs(basis.overlap(final_state)) ** 2
                current_probs.append(prob)
            self.iter_history["prob"] = np.vstack([self.iter_history["prob"], current_probs])
            
            return cost_val
        
        # Initial params
        init_params = np.random.uniform(0, np.pi, self.paramlength * self.p)
        
        result = minimize(
            fun=cost_with_history,
            x0=init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "disp": True}
        )
        
        self.optimal_params = result.x
        self.final_state = self.qaoa_circuit(self.optimal_params, initial_state)
        self.final_cost = result.fun
        self.final_obj = -self.final_cost  # Maximized objective
        
        print(f"\nOptimization complete: Optimal params {self.optimal_params.round(4)}, Obj={self.final_obj:.4f}")
        return {"params": self.optimal_params, "obj": self.final_obj, "state": self.final_state}
    
    def get_fock_probs(self, state: qt.Qobj) -> np.ndarray:
        """P(n) for all n in [0,N)^d."""
        probs = np.zeros((self.N,) * self.num_modes)
        for ns in product(range(self.N), repeat=self.num_modes):
            basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
            probs[ns] = abs(basis.overlap(state)) ** 2
        return probs
    
    def plot_results(self, save_path: str = "figs/qaoa_ip_results.svg") -> None:
        """Plot iteration history, costs, probs, objectives."""
        final_probs = self.get_fock_probs(self.final_state)
        final_tracked_probs = [final_probs[ns] for ns in self.tracked_states]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Subplot 1: Tracked state probs over iterations
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.tracked_states)))
        for i, label in enumerate(self.tracked_labels):
            axes[0, 0].plot(self.iter_history["iter"], self.iter_history["prob"][:, i],
                            linewidth=2.5, label=label, color=colors[i], marker="o", markersize=3)
        axes[0, 0].set_xlabel("Optimization Iteration")
        axes[0, 0].set_ylabel("State Probability")
        axes[0, 0].set_title(f"QAOA Iteration: Tracked State Probabilities (p={self.p})")
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(alpha=0.3)
        
        # Subplot 2: Cost convergence
        axes[0, 1].plot(self.iter_history["iter"], self.iter_history["cost"],
                        linewidth=3, color="red", marker="s", markersize=4)
        axes[0, 1].set_xlabel("Optimization Iteration")
        axes[0, 1].set_ylabel("<H_C>")
        axes[0, 1].set_title("Cost Function Convergence")
        axes[0, 1].grid(alpha=0.3)
        
        # Subplot 3: Final tracked probs bar
        bars = axes[1, 0].bar(range(len(final_tracked_probs)), final_tracked_probs, color=colors)
        for bar, prob in zip(bars, final_tracked_probs):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f"{prob:.3f}", ha="center", va="bottom", fontsize=10)
        axes[1, 0].set_xlabel("Tracked States")
        axes[1, 0].set_ylabel("Final Probability")
        axes[1, 0].set_title("Final State Probabilities")
        axes[1, 0].set_xticks(range(len(final_tracked_probs)))
        axes[1, 0].set_xticklabels([lbl.split(" (")[0] for lbl in self.tracked_labels], rotation=45, ha="right")
        axes[1, 0].grid(axis="y", alpha=0.3)
        
        # Subplot 4: Objective over iterations
        obj_history = [-cost for cost in self.iter_history["cost"]]
        max_obj = max([sum(self.c * ns) for ns in self.feasible_states])
        axes[1, 1].plot(self.iter_history["iter"], obj_history,
                        linewidth=3, color="green", marker="^", markersize=4)
        axes[1, 1].axhline(y=max_obj, color="orange", linestyle="--", linewidth=2, label=f"Max Obj ({max_obj})")
        axes[1, 1].set_xlabel("Optimization Iteration")
        axes[1, 1].set_ylabel("Objective <c^T x>")
        axes[1, 1].set_title("Objective Improvement")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def print_summary(self) -> None:
        """Print optimization summary and top states."""
        print(f"\n=== IP Solution Summary ===")
        print(f"Objective: {self.final_obj:.4f} (max possible: {max(sum(self.c * ns) for ns in self.feasible_states):.4f})")
        print(f"Improvement: {self.final_obj + self.iter_history['cost'][0]:.4f}")
        
        final_probs = self.get_fock_probs(self.final_state)
        top_feasible = sorted(
            [(ns, final_probs[ns], sum(self.c * ns))
             for ns in self.feasible_states if final_probs[ns] > 1e-3],
            key=lambda x: x[1], reverse=True
        )
        print(f"\nTop 3 feasible states by probability:")
        for i, (ns, prob, obj) in enumerate(top_feasible[:3]):
            print(f"  {i+1}: |{','.join(map(str, ns))}⟩ → P={prob:.4f}, Obj={obj:.4f}")

    def _build_constraint_violation_operator(self) -> qt.Qobj:
        """Build violation operator V."""
        if self.constraint_type == "linear":
            m_constraints = self.A.shape[0]
            V = 0 * self.n_ops[0]
            for j in range(m_constraints):
                C_j = sum(self.A[j, i] * self.n_ops[i] for i in range(self.num_modes))
                V += (C_j - self.b[j]) ** 2
        else:  # quadratic
            C = 0 * self.n_ops[0]
            for i in range(self.num_modes):
                for j in range(self.num_modes):
                    C += self.A[i,j] * self.n_ops[i] * self.n_ops[j]
            V = (C - self.b[0]) ** 2
        return V
    

    def _get_lindblad_operators(self, error_config: Dict) -> List[qt.Qobj]:
        """Return Lindblad operators L_k for a given error config."""
        error_type = error_config.get('type', '')
        mode = error_config.get('mode', 0)
        rate = error_config.get('rate', 1.0)
        n_th = error_config.get('n_th', 0.5)
        chi = error_config.get('chi', 0.1)
        eta = error_config.get('eta', 0.5)
        imbalance_rate = error_config.get('imbalance_rate', 0.1)
        other_mode = error_config.get('other_mode', 1)  # For cross-mode

        Ls = []

        if error_type == 'photon_loss':
            Ls.append(np.sqrt(rate) * self.a_ops[mode])
        elif error_type == 'photon_gain':
            Ls.append(np.sqrt(rate) * self.ad_ops[mode])
        elif error_type == 'thermal':
            L_down = np.sqrt(rate * (n_th + 1)) * self.a_ops[mode]
            L_up = np.sqrt(rate * n_th) * self.ad_ops[mode]
            Ls.extend([L_down, L_up])
        elif error_type == 'cross_mode_unbalanced':
            # Beam-splitter term + imbalance
            L_bs = np.sqrt(eta) * (self.ad_ops[mode] * self.a_ops[other_mode] + self.a_ops[mode] * self.ad_ops[other_mode])
            L_imbal = np.sqrt(imbalance_rate) * self.a_ops[mode]
            Ls.extend([L_bs, L_imbal])
        elif error_type == 'kerr_loss':
            # Kerr is coherent, added to H; loss as L
            Ls.append(np.sqrt(imbalance_rate) * self.a_ops[mode])  # Reuse imbalance_rate as loss rate
            # Note: chi added to H in simulation call

        return Ls

    def simulate_errors(
        self,
        error_configs: List[Dict],
        tlist: np.ndarray,
        H_evol: Optional[qt.Qobj] = None,
        initial_state: Optional[qt.Qobj] = None,
        plot: bool = True,
        save_path: str = "figs/error_simulation.svg"
    ) -> Dict[str, np.ndarray]:
        """
        Simulate subspace confinement under noisy evolution for multiple error configurations.

        The constraint violation is quantified by the expectation value of the operator
        \[
        \hat{V} = \sum_{j=1}^m \left( \sum_{i=1}^d A_{j,i} \hat{n}_i - b_j \right)^2,
        \]
        where \(\langle \hat{V} \rangle = 0\) indicates perfect confinement in the feasible subspace \(\{ | \mathbf{n} \rangle \mid A \mathbf{n} = \mathbf{b} \}\).

        Args:
            error_configs: List of dicts, each specifying an error, e.g.,
                           [{'type': 'photon_loss', 'mode': 0, 'rate': 1.0},
                            {'type': 'thermal', 'mode': 0, 'rate': 1.0, 'n_th': 0.5}, ...]
            tlist: Time array for evolution.
            H_evol: Coherent Hamiltonian for evolution (default: self.H_M).
            initial_state: Initial state (default: uniform superposition over feasible states).
            plot: Whether to plot violation vs. time.
            save_path: Path to save plot.

        Returns:
            Dict of {error_label: violation(t)} arrays.
        """
        if H_evol is None:
            H_evol = self.H_M
        if initial_state is None:
            # Uniform superposition over feasible states
            psi_list = [qt.tensor(*[qt.fock(self.N, n) for n in ns]) for ns in self.feasible_states]
            initial_state = sum(psi_list) / np.sqrt(len(psi_list))
        rho0 = initial_state * initial_state.dag()

        V = self._build_constraint_violation_operator()  # Violation operator

        violations = {}  # {label: <V>(t)}

        for config in error_configs:
            label = f"{config['type']} (mode {config.get('mode', 0)})"
            Ls = self._get_lindblad_operators(config)

            # For Kerr, add H_kerr to H_evol
            H_total = H_evol
            if config.get('type') == 'kerr_loss':
                chi = config.get('chi', 0.1)
                H_kerr = chi * self.n_ops[config.get('mode', 0)] * (self.n_ops[config.get('mode', 0)] - 1) / 2
                H_total += H_kerr

            # Evolve under Lindblad ME
            result = qt.mesolve(H_total, rho0, tlist, Ls)
            expects = qt.expect(V, result.states)
            violations[label] = expects

        if plot:
            plt.figure(figsize=(10, 6))
            for label, viol_t in violations.items():
                plt.plot(tlist, viol_t, label=label, linewidth=2)
            # Ideal (no noise)
            result_ideal = qt.mesolve(H_evol, rho0, tlist, [])
            viol_ideal = qt.expect(V, result_ideal.states)
            plt.plot(tlist, viol_ideal, 'k--', label='Ideal (No Error)', linewidth=2)
            plt.xlabel('Time $t$')
            plt.ylabel(r'$\langle \hat{V} \rangle$')
            plt.title(r'Constraint Violation $\langle \hat{V} \rangle$ Under Errors')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.yscale('log')  # Log scale for small violations
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

        return violations

# Example usage for original problem
if __name__ == "__main__":
    A = [[2, 1, 1, 1],[1, 2, 1, 1]]
    b = [6,4]
    c = [1, -1, 1, 2]
    
    solver = BosonicQAOAIPSolver(A, b, c, N=7, p=2,circuit_type="multi_beta")
    ## Hamiltonian of solver
    print("Driver Hamiltonian H_M:")
    print(solver.latex_label_H_M)
    print(solver.H_M)
    result = solver.optimize()
    solver.plot_results(save_path="figs/qaoa_ip_multi_beta.svg")
    solver.print_summary()

    # Example error simulation
    error_configs = [
        {'type': 'photon_loss', 'mode': 0, 'rate': 1.0},
        {'type': 'photon_gain', 'mode': 0, 'rate': 1.0},
        {'type': 'thermal', 'mode': 0, 'rate': 1.0, 'n_th': 0.5},
        {'type': 'cross_mode_unbalanced', 'mode': 0, 'other_mode': 1, 'eta': 0.5, 'imbalance_rate': 0.1},
        {'type': 'kerr_loss', 'mode': 0, 'chi': 0.1, 'imbalance_rate': 0.05}
    ]
    tlist = np.linspace(0, 0.1, 50)
    violations = solver.simulate_errors(error_configs, tlist, H_evol=solver.H_M)