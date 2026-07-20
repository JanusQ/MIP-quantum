import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import minimize
import sympy as sp
from typing import List, Tuple, Optional, Union, Dict
from scipy.linalg import eigh


class BosonicQAOAIPSolver:
    """
    Bosonic QAOA solver for arbitrary non-negative integer linear programming (IP):
    max c^T x s.t. A x = b, x >= 0 integer.
    
    Encoding: x_i <-> n_i (photon number in mode i).
    Constraint subspace S_c: { |n> | A n = b }.
    Driver H_M: sum_u (O_u + O_u^dagger) over integer nullspace basis of A.
    Cost H_C = - sum c_i n_i.
    
    Args:
        A: Constraint matrix (m x d, integer coeffs).
        b: RHS vector (m, integer targets).
        c: Objective coeffs (d, maximize c^T x).
        N: Fock truncation per mode.
        p: QAOA layers.
        num_modes: d (inferred from A/c if not given).
        g: Driver coupling strength.
        maxiter: Optimization iterations.
        seed: Random seed.
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
        circuit_type: str = "beta_gamma"  # or "multi_beta"
    ):
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
        if self.A.shape[1] != self.num_modes or self.c.shape[0] != self.num_modes:
            raise ValueError("Dimensions mismatch: A (m x d), b (m), c (d).")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError("A rows != b length.")
        
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
        # self.initial_state = np.array([2,0,3,3])
        self.tracked_states = sorted(self.feasible_states, key=lambda ns: sum(self.c * ns), reverse=True)[:6]+[self.initial_state]
        self.tracked_labels = [f"|{','.join(map(str, ns))}⟩ (obj={sum(self.c * ns):.1f})" for ns in self.tracked_states]
        
        print(f"Initialized BosonicQAOAIPSolver: {self.num_modes} modes, {len(self.null_space_basis)} null vectors.")
        print(f"Constraints: A x = b (shape {self.A.shape}). Objective: max {self.c} · x.")
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
            # Make primitive: gcd of components
            int_vec = np.array(int_vec).flatten().astype(int)
            gcd = np.gcd.reduce(int_vec)
            if gcd != 0:
                int_vec = int_vec // gcd
            # Flip sign if first non-zero is negative
            if int_vec[int(np.nonzero(int_vec)[0][0])] < 0:
                int_vec = -int_vec
            int_vecs.append(int_vec)
        
        # Remove duplicates (if any)
        unique_vecs = set(tuple(v) for v in int_vecs)
        unique_vecs = np.array(list(unique_vecs))
        
        # add a new vector for circle loop
        # add first vector and last vector to make a circle loop， then gcd to make it primitive
        if len(unique_vecs)>1:
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
            if new_vec[int(np.nonzero(new_vec)[0][0])] < 0:
                new_vec = -new_vec
            unique_vecs = np.vstack([unique_vecs, new_vec])
        
        ## add reverse direction
        # reversed_vecs = -unique_vecs
        # unique_vecs = np.vstack([unique_vecs, reversed_vecs])
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
    
    def _build_driver_hamiltonian(self) -> qt.Qobj:
        """H_M = g sum_u (O_u + O_u^dagger)."""
        H_M = 0 * self.a_ops[0]
        for u in self.null_space_basis:
            O_u = qt.qeye(1)
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** u[i])
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M += self.g * (O_u + O_u.dag())
        return H_M
    
    def _build_seperate_driver_hamiltonian(self) -> List[qt.Qobj]:
        """H_M = g sum_u (O_u + O_u^dagger)."""
        Hds = []
        for u in self.null_space_basis:
            H_M = 0 * self.a_ops[0]
            O_u = 1
            for i in range(self.num_modes):
                if u[i] > 0:
                    O_u = O_u * (self.ad_ops[i] ** int(u[i]))
                elif u[i] < 0:
                    O_u = O_u * (self.a_ops[i] ** abs(u[i]))
            H_M += self.g * (O_u + O_u.dag())
            Hds.append(H_M)
        return Hds
    
    def _find_feasible_states(self) -> List[Tuple[int, ...]]:
        """Enumerate feasible n in [0,N)^d with A n = b (integer)."""
        feasible = []
        for ns_tuple in product(range(self.N), repeat=self.num_modes):
            ns = np.array(ns_tuple)
            if np.allclose(self.A @ ns, self.b):
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
    
    def optimize(
        self,
        mode: str = 'ideal',
        initial_state: Optional[qt.Qobj] = None,
        noise_config: Optional[Dict] = None,
        gse_K: int = 2,
        **kwargs  # Pass to minimize (e.g., maxiter)
    ) -> dict:
        """
        Optimize QAOA with optional noise and GSE mitigation.

        Modes:
            'ideal': Noiseless evaluation.
            'noisy': Noisy circuit (noise after each layer).
            'gse': Noisy circuit + sequential GSE (mitigate V, then <H_C>).

        Args:
            mode: Optimization mode ('ideal', 'noisy', 'gse').
            initial_state: Initial state.
            noise_config: Dict e.g., {'type': 'amplitude_damping', 'rate': 0.05}.
            gse_K: GSE power subspace order (for 'gse' mode).
            **kwargs: Options for scipy.optimize.minimize.

        Returns:
            Dict with 'params', 'obj', 'state' (noisy rho for 'noisy'/'gse'), 'history' (dict of iter/cost/prob).
        """
        valid_modes = ['ideal', 'noisy', 'gse']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}.")
        if mode in ['noisy', 'gse'] and noise_config is None:
            raise ValueError("noise_config required for 'noisy' or 'gse' modes.")

        if initial_state is None:
            initial_state = self.create_initial_state(superposition=True)
        self.mode = mode  # Store for history

        # History dict
        self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
        
        def cost_with_history(params: np.ndarray) -> float:
            iter_idx = len(self.iter_history["iter"])
            self.iter_history["iter"].append(iter_idx + 1)
            
            # Evaluate circuit (ideal or noisy)
            if mode == 'ideal':
                final_state = self.qaoa_circuit(params, initial_state)
            else:
                final_state = self._evaluate_noisy_circuit(params, initial_state, noise_config)
            
            # Compute cost (raw or GSE-mitigated)
            if mode == 'gse':
                V = self._build_constraint_violation_operator()
                _, _, rho_em_v = self.mitigate_gse_violation(final_state, V, K=gse_K)
                cost_val = (rho_em_v * self.H_C).tr().real
            else:
                cost_val = qt.expect(self.H_C, final_state)
            
            self.iter_history["cost"].append(cost_val)
            
            # Track probs (on final_state for simplicity; use rho_em_v for 'gse' if desired)
            if mode == 'gse':
                track_state = rho_em_v
            else:
                track_state = final_state
            current_probs = []
            for ns in self.tracked_states:
                basis = qt.tensor(*[qt.fock(self.N, n) for n in ns])
                prob = abs(basis.overlap(track_state)) ** 2
                current_probs.append(prob)
            self.iter_history["prob"] = np.vstack([self.iter_history["prob"], current_probs])
            
            return cost_val

        # Initial params
        init_params = np.random.uniform(0, np.pi, self.paramlength * self.p)
        
        result = minimize(
            fun=cost_with_history,
            x0=init_params,
            method="COBYLA",
            options={**{"maxiter": self.maxiter, "disp": True}, **kwargs}
        )
        
        self.optimal_params = result.x
        if mode == 'ideal':
            self.final_state = self.qaoa_circuit(self.optimal_params, initial_state)
        else:
            self.final_state = self._evaluate_noisy_circuit(self.optimal_params, initial_state, noise_config)
            if mode == 'gse':
                V = self._build_constraint_violation_operator()
                _, _, self.final_state = self.mitigate_gse_violation(self.final_state, V, K=gse_K)
        self.final_cost = result.fun
        self.final_obj = -self.final_cost
        
        print(f"\n{mode.upper()} Optimization complete: Obj={self.final_obj:.4f}")
        return {"params": self.optimal_params, "obj": self.final_obj, "state": self.final_state, "history": self.iter_history}

    def _evaluate_noisy_circuit(self, params: np.ndarray, initial_state: qt.Qobj, noise_config: Dict) -> qt.Qobj:
        """Evaluate noisy QAOA circuit with noise after each unitary layer."""
        rho = initial_state * initial_state.dag()
        if self.circuit_type == "beta_gamma":
            for layer in range(self.p):
                gamma = params[2 * layer]
                beta = params[2 * layer + 1]
                # Cost layer
                U_C = (-1j * self.H_C * gamma).expm()
                rho = U_C * rho * U_C.dag()
                rho = self._apply_noise(rho, noise_config)
                # Mixer layer
                U_M = (-1j * self.H_M * beta).expm()
                rho = U_M * rho * U_M.dag()
                rho = self._apply_noise(rho, noise_config)
        elif self.circuit_type == "multi_beta":
            betalength = len(self.null_space_basis)
            for layer in range(self.p):
                for i in range(betalength):
                    beta = params[layer * betalength + i]
                    U_di = (-1j * self.H_ds[i] * beta).expm()
                    rho = U_di * rho * U_di.dag()
                    rho = self._apply_noise(rho, noise_config)
        elif self.circuit_type == "multi_beta_oneH":
            betalength = len(self.null_space_basis)
            for layer in range(self.p):
                betas = params[layer * betalength : (layer + 1) * betalength]
                H_d_layer = sum(betas[i] * self.H_ds[i] for i in range(betalength))
                U_d = (-1j * H_d_layer).expm()
                rho = U_d * rho * U_d.dag()
                rho = self._apply_noise(rho, noise_config)
        else:
            raise ValueError(f"Noisy evaluation not supported for circuit_type '{self.circuit_type}'.")
        return rho

    def plot_optimization_comparison(
        self,
        modes: List[str],
        noise_configs: List[Dict],
        gse_K: int = 2,
        save_path: str = "figs/optimization_comparison.svg"
    ) -> None:
        """
        Run optimizations for multiple modes and noise configurations, plotting objective convergence.

        Args:
            modes: List of modes to include (e.g., ['ideal', 'noisy', 'gse']).
            noise_configs: List of noise configurations (Dicts) for 'noisy' and 'gse' modes.
            gse_K: GSE order for 'gse'.
        """
        histories = {}
        initial_state_opt = self.create_initial_state(superposition=True)
        
        # Run ideal if requested
        if 'ideal' in modes:
            self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
            _ = self.optimize(mode='ideal', initial_state=initial_state_opt)
            histories['Ideal'] = self.iter_history
        
        # For each noise config, run noisy and/or gse if requested
        for config in noise_configs:
            label = self._get_config_label(config)
            if 'noisy' in modes:
                self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
                _ = self.optimize(mode='noisy', initial_state=initial_state_opt, noise_config=config)
                histories[f'Noisy - {label}'] = self.iter_history
            if 'gse' in modes:
                self.iter_history = {"iter": [], "cost": [], "prob": np.zeros((0, len(self.tracked_states)))}
                _ = self.optimize(mode='gse', initial_state=initial_state_opt, noise_config=config, gse_K=gse_K)
                histories[f'GSE - {label}'] = self.iter_history
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        all_keys = list(histories.keys())
        num_lines = len(all_keys)
        colors = plt.cm.tab10(np.linspace(0, 1, num_lines))
        
        max_obj = max([sum(self.c * ns) for ns in self.feasible_states])
        
        for idx, key in enumerate(all_keys):
            hist = histories[key]
            obj_hist = [-c for c in hist["cost"]]
            color = colors[idx]
            # Determine linestyle based on mode prefix
            if key == 'Ideal':
                ls = '-'
            elif key.startswith('Noisy'):
                ls = '--'
            elif key.startswith('GSE'):
                ls = ':'
            else:
                ls = '-'
            ax.plot(hist["iter"], obj_hist, label=key, color=color, linewidth=2, linestyle=ls)
        
        ax.axhline(y=max_obj, color='gray', ls=':', alpha=0.7, label=f'Classical Max ({max_obj:.2f})')
        
        ax.set_xlabel("Optimization Iteration")
        ax.set_ylabel("Objective $\langle \mathbf{c}^\top \mathbf{x} \rangle$")
        ax.set_title("QAOA Optimization Convergence: Ideal vs. Noisy vs. GSE-Mitigated")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def _get_config_label(self, config: Dict) -> str:
        """Extract a label from noise config."""
        t = config.get('type', 'unknown')
        extras = []
        if 'rate' in config and config['rate'] is not None:
            extras.append(f"r={config['rate']}")
        if 'mode' in config and config['mode'] is not None:
            extras.append(f"m={config['mode']}")
        return t + " " + " ".join(extras) if extras else t
        
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

    def _build_identity(self) -> qt.Qobj:
        """Build identity operator for multi-mode Hilbert space."""
        ops = [qt.qeye(self.N) for _ in range(self.num_modes)]
        return qt.tensor(ops)

    def mitigate_gse_expect(self, rho_noisy: qt.Qobj, O: qt.Qobj, subspace_type: str = 'power', K: int = 2, A: Optional[qt.Qobj] = None) -> Tuple[float, float, qt.Qobj]:
        """
        Mitigate expectation value <O> using Generalized Quantum Subspace Expansion (GSE).

        Constructs the mitigated state via the power subspace spanned by {I, ρ, ρ², ..., ρᴷ},
        solving the generalized eigenvalue problem for the lowest "energy" under O as the effective Hamiltonian.

        Note: Computationally intensive for large Hilbert dimensions; suitable for small N or num_modes.

        Returns:
            Tuple (raw_expectation, gse_mitigated_expectation, mitigated_rho)
        """
        if A is None:
            A = self._build_identity()

        # Build power subspace bases σ_i = ρ^{i} for i = 0 to K
        sigmas = []
        current = self._build_identity()
        rho_current = current
        for i in range(K + 1):
            sigmas.append(rho_current)
            rho_current = rho_current * rho_noisy

        d_s = len(sigmas)
        S = np.zeros((d_s, d_s), dtype=complex)
        O_mat = np.zeros((d_s, d_s), dtype=complex)  # Effective matrix for O

        for i in range(d_s):
            sigma_i_dag = sigmas[i].dag()
            for j in range(d_s):
                temp = sigma_i_dag * A * sigmas[j]
                S[i, j] = temp.tr().real
                O_mat[i, j] = (temp * O).tr()

        # Solve generalized eigenvalue problem: O α = E S α, select minimal E
        evals, evecs = eigh(O_mat, S)
        idx_min = np.argmin(evals.real)
        alpha = evecs[:, idx_min]
        norm = np.sqrt(np.real(alpha.conj().T @ S @ alpha))
        if norm > 1e-10:
            alpha = alpha / norm

        rho_mit = sum(alpha[i] * sigmas[i] for i in range(d_s))
        o_raw = (rho_noisy * O).tr().real
        o_gse = np.real(alpha.conj().T @ O_mat @ alpha)

        return o_raw, o_gse, rho_mit

    def _apply_noise(self, rho: qt.Qobj, noise_config: Dict) -> qt.Qobj:
        """Apply noise channel to density operator ρ."""
        dt = 0.01  # Small time step for approximation

        Ls = self._get_lindblad_operators(noise_config)

        H_noise = 0 * rho
        noise_type = noise_config.get('type', '')
        if noise_type == 'kerr_loss':
            chi = noise_config.get('chi', 0.1)
            mode = noise_config.get('mode', 0)
            H_kerr = chi * self.n_ops[mode] * (self.n_ops[mode] - 1) / 2
            H_noise += H_kerr

        times = [0, dt]
        result = qt.mesolve(H_noise, rho, times, Ls)
        return result.states[-1]

    def simulate_noisy_qaoa(self, params: np.ndarray, noise_config: Dict, initial_state: Optional[qt.Qobj] = None) -> qt.Qobj:
        """
        Simulate noisy QAOA circuit with noise after each layer.

        Args:
            params: Optimized parameters from optimize().
            noise_config: Dict e.g., {'type': 'amplitude_damping', 'rate': 0.05}.
            initial_state: Initial state (default: self.create_initial_state()).

        Returns:
            Noisy final density operator ρ.
        """
        if initial_state is None:
            initial_state = self.create_initial_state(superposition=True)
        return self._evaluate_noisy_circuit(params, initial_state, noise_config)

    def _build_constraint_violation_operator(self) -> qt.Qobj:
        """Build violation operator V = \sum_j ( \sum_i A_{j,i} \hat{n}_i - b_j )^2."""
        m_constraints = self.A.shape[0]
        V = 0 * self.n_ops[0]
        for j in range(m_constraints):
            C_j = sum(self.A[j, i] * self.n_ops[i] for i in range(self.num_modes))
            V += (C_j - self.b[j]) ** 2
        return V

    def _get_lindblad_operators(self, error_config: Dict) -> List[qt.Qobj]:
        """Return Lindblad operators L_k for a given error config."""
        error_type = error_config.get('type', '')
        mode = error_config.get('mode', 0)
        rate = error_config.get('rate', 1.0)
        n_th = error_config.get('n_th', 0.5)
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
        elif error_type == 'amplitude_damping':
            # Amplitude damping on all modes (zero temperature)
            Ls = [np.sqrt(rate) * a for a in self.a_ops]
        elif error_type == 'dephasing':
            # Dephasing on number basis
            Ls = [np.sqrt(rate) * n for n in self.n_ops]
        else:
            raise ValueError(f"Unsupported noise type: {error_type}")

        return Ls

    def mitigate_gse_violation(self, rho_noisy: qt.Qobj, V: qt.Qobj, K: int = 2, A: Optional[qt.Qobj] = None) -> Tuple[float, float, qt.Qobj]:
        """
        Mitigate constraint violation <V> using GSE on noisy rho.

        Similar to mitigate_gse_expect, but specialized for V.
        Returns (raw <V>, GSE <V>, mitigated_rho).
        """
        return self.mitigate_gse_expect(rho_noisy, V, subspace_type='power', K=K, A=A)

    def simulate_errors(
        self,
        error_configs: List[Dict],
        tlist: np.ndarray,
        H_evol: Optional[qt.Qobj] = None,
        initial_state: Optional[qt.Qobj] = None,
        gse_K: int = 2,
        plot: bool = True,
        save_path: str = "figs/error_simulation_gse.svg"
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Simulate subspace confinement under noisy evolution, with GSE mitigation.

        Computes both raw <V>(t) and GSE-mitigated <V_gse>(t) for each config.

        Args:
            ... (same as before)
            gse_K: Order K for power subspace {I, ρ, ..., ρ^K}.
        
        Returns:
            Dict of {error_label: (raw_viol(t), gse_viol(t))} tuples.
        """
        if H_evol is None:
            H_evol = self.H_M
        if initial_state is None:
            initial_state = self.create_initial_state()
        rho0 = initial_state * initial_state.dag()

        V = self._build_constraint_violation_operator()
        I = self._build_identity()
        A_gse = I  # Default A = I

        results = {}  # {label: (raw_expects, gse_expects)}

        for config in error_configs:
            label = f"{config['type']} (mode {config.get('mode', 0)})"
            Ls = self._get_lindblad_operators(config)

            H_total = H_evol
            if config.get('type') == 'kerr_loss':
                chi = config.get('chi', 0.1)
                mode = config.get('mode', 0)
                H_kerr = chi * self.n_ops[mode] * (self.n_ops[mode] - 1) / 2
                H_total += H_kerr

            # Evolve
            result = qt.mesolve(H_total, rho0, tlist, Ls)
            raw_expects = qt.expect(V, result.states)

            # GSE mitigation for each t
            gse_expects = np.zeros_like(raw_expects)
            for idx, rho_t in enumerate(result.states):
                raw, gse_val, _ = self.mitigate_gse_violation(rho_t, V, K=gse_K, A=A_gse)
                gse_expects[idx] = gse_val

            results[label] = (raw_expects, gse_expects)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10(np.linspace(0, 1, len(error_configs)))
            for idx, (label, (raw_t, gse_t)) in enumerate(results.items()):
                color = colors[idx]
                ax.plot(tlist, raw_t, label=f"{label} (Raw)", color=color, linewidth=2, linestyle='-')
                ax.plot(tlist, gse_t, label=f"{label} (GSE)", color=color, linewidth=2, linestyle='--')

            # Ideal
            result_ideal = qt.mesolve(H_evol, rho0, tlist, [])
            viol_ideal = qt.expect(V, result_ideal.states)
            _, gse_ideal, _ = self.mitigate_gse_violation(result_ideal.states[0], V, K=gse_K, A=A_gse)  # Constant
            ax.plot(tlist, viol_ideal, 'k-', label='Ideal (Raw)', linewidth=2)
            ax.plot(tlist, np.full_like(tlist, gse_ideal), 'k--', label='Ideal (GSE)', linewidth=2)

            ax.set_xlabel('Time $t$')
            ax.set_ylabel(r'$\langle \hat{V} \rangle$')
            ax.set_title(r'Constraint Violation $\langle \hat{V} \rangle$ Under Errors (Raw vs. GSE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()

        return results


# Import the full BosonicQAOAIPSolver class definition here
# (Paste the complete class code from previous responses, including all methods)

if __name__ == "__main__":
    # Step 1: Define problem instance (small for demo: max x1 - x2 s.t. x1 + x2 = 1)
    A = np.array([[1,-1,1],[1,2,-1]])  # Constraint matrix
    b = np.array([3,4])        # RHS vector
    c = np.array([1, 2, 1])  # Objective coefficients
    N_trunc = 5              # Fock truncation (small for efficiency)
    p_layers = 2             # QAOA layers
    g_driver = 1.0           # Driver strength

    # Step 2: Initialize solver
    solver = BosonicQAOAIPSolver(
        A=A, b=b, c=c, N=N_trunc, p=p_layers, g=g_driver,
        circuit_type="beta_gamma", maxiter=50, seed=42
    )
    noise_cfgs = [{'type': 'photon_loss', 'rate': 0.05, 'mode': 0}, {'type': 'dephasing', 'rate': 0.03}]
    solver.plot_optimization_comparison(modes=['ideal', 'noisy', 'gse'], noise_configs=noise_cfgs, gse_K=2)