import numpy as np
import qiskit
import c2qa
from typing import List, Dict, Callable
from gate import SQR, cv_R, cv_CRi_dag, add_RR
class AQC_MIP_Solver:
    """
    Solves Mixed-Integer Programming (MIP) problems using simulated
    Adiabatic Quantum Computation (AQC) with the c2qa library.

    This class translates a MIP problem into a problem Hamiltonian (Hp) and
    simulates the adiabatic evolution from an initial mixing Hamiltonian (Hm)
    to find the ground state of Hp, which encodes the solution.
    """

    def __init__(
        self,
        num_qumodes: int,
        num_qubits_per_qumode: int,
        hamiltonian_terms: List[Dict],
        total_time: float = 50.0,
        num_trotter_steps: int = 100,
        initial_squeezing: float = 2.0,
    ):
        """
        Initializes the AQC solver.

        Args:
            num_qumodes (int): The total number of qumodes required for the problem
                               (variables + slack variables).
            num_qubits_per_qumode (int): The number of qubits to represent each qumode,
                                         determining the Fock space cutoff (2^n).
            hamiltonian_terms (List[Dict]): A list of dictionaries, each describing a
                                           term in the problem Hamiltonian.
                                           Example: [{'type': 'n', 'qumode': 0, 'coeff': -1.0},
                                                     {'type': 'n_squared', 'qumode': 1, 'coeff': 0.5}]
            total_time (float): The total evolution time T for the adiabatic schedule.
            num_trotter_steps (int): The number of steps for the Trotterization.
            initial_squeezing (float): The squeezing parameter 'r' for the initial state.
        """
        self.num_qumodes = num_qumodes
        self.num_qubits_per_qumode = num_qubits_per_qumode
        self.fock_cutoff = 2**num_qubits_per_qumode
        self.hamiltonian_terms = hamiltonian_terms
        self.total_time = total_time
        self.num_trotter_steps = num_trotter_steps
        self.dt = total_time / num_trotter_steps
        self.initial_squeezing = initial_squeezing
        self.ancqubit = qiskit.QuantumRegister(1, name="ancqubit")
        self.qmr = c2qa.QumodeRegister(num_qumodes, num_qubits_per_qumode, name="qmr")
        # ClassicalRegister size must match total number of qubits, not qumodes
        self.cr = qiskit.ClassicalRegister(self.qmr.size, name="cr")
        self.circuit = c2qa.CVCircuit(self.qmr,self.ancqubit, self.cr)

    def _apply_hamiltonian_gates(self, hamiltonian_schedule: float):
        """
        Applies gates corresponding to the problem Hamiltonian (Hp) for one Trotter step.

        Args:
            hamiltonian_schedule (float): The schedule parameter s(t) for Hp.
        """
        for term in self.hamiltonian_terms:
            op_type = term.get("type")
            qumode_idx = term.get("qumode")
            qumode_idx_2 = term.get("qumode2") # For two-qumode gates
            coeff = term.get("coeff")
            angle = coeff * hamiltonian_schedule * self.dt

            if op_type == "n":
                # Term: c * n -> Gate: exp(-i * c * n * dt) -> Phase space rotation
                self.circuit.cv_r(-angle, self.qmr[qumode_idx])
            
            elif op_type == "n_squared":
                # FIX: Use SNAP gate correctly by providing a list of phases.
                # The SNAP gate applies a phase theta_k to each Fock state |k>.
                # For a Hamiltonian term c*n^2, the unitary is exp(-i * c * n^2 * dt).
                # The phase for state |k> is -c * k^2 * dt, which is -angle * k^2.
                # We provide a list of phases for all Fock states up to the cutoff.
                phases = [-angle * (k**2) for k in range(self.fock_cutoff)]
                self.circuit.cv_snap(phases,list(range(self.fock_cutoff)), self.qmr[qumode_idx])

            elif op_type == "n1n2":
                 # FIX: Synthesize cross-Kerr with the more direct cv_c_r gate.
                 # Term: c * n1*n2 -> Gate: exp(-i * c * n1*n2 * dt)
                 # This is equivalent to a controlled phase rotation, where the rotation
                 # angle on qumode 2 depends on the photon number in qumode 1.
                #  self.circuit.cv_c_r(-angle, self.qmr[qumode_idx], self.qmr[qumode_idx_2])
                add_RR(self.circuit, self.ancqubit, self.qmr[qumode_idx], self.qmr[qumode_idx_2], 2*angle, self.fock_cutoff)

            else:
                print(f"Warning: Hamiltonian term type '{op_type}' is not recognized.")

    def _apply_mixing_hamiltonian_gates(self, mixer_schedule: float):
        """
        Applies gates for the mixing Hamiltonian Hm = sum(p_i^2) for one Trotter step.
        This corresponds to a squeezing operation.

        Args:
            mixer_schedule (float): The schedule parameter (1 - s(t)) for Hm.
        """
        angle = mixer_schedule * self.dt
        for i in range(self.num_qumodes):
            # Term: p^2 -> Gate: exp(-i * p^2 * dt) -> Squeezing gate
            # Squeezing gate in c2qa is defined as exp(0.5 * (r*a^2 - r*a_dag^2))
            # exp(-i * p^2 * dt) corresponds to a squeezing parameter r = -2 * angle
            self.circuit.cv_sq(angle, self.qmr[i])

    def build_circuit(self):
        """
        Constructs the full AQC circuit.
        """
        print("Building AQC circuit...")
        # 1. Prepare initial state: ground state of Hm (approximated by squeezed vacuum)
        for i in range(self.num_qumodes):
            self.circuit.cv_sq(self.initial_squeezing, self.qmr[i])

        self.circuit.barrier()

        # 2. Adiabatic Evolution (Trotterized)
        for step in range(self.num_trotter_steps):
            # Linear schedule s(t) = t / T
            schedule = (step + 1) / self.num_trotter_steps

            # Apply problem Hamiltonian gates
            self._apply_hamiltonian_gates(hamiltonian_schedule=schedule)

            # Apply mixing Hamiltonian gates
            self._apply_mixing_hamiltonian_gates(mixer_schedule=(1.0 - schedule))

            self.circuit.barrier()

        # 3. Add measurement
        self.circuit.cv_measure(self.qmr, self.cr)
        print("Circuit build complete.")


    def solve(self, shots: int = 2048):
        """
        Simulates the AQC circuit and returns the results.

        Args:
            shots (int): The number of simulation shots to run.

        Returns:
            dict: A dictionary of measurement counts for each Fock state.
        """
        if not self.circuit.data:
            self.build_circuit()

        print(f"Simulating circuit with {shots} shots...")
        # Use the 'c2qa_fock' simulator for direct Fock state simulation
        _, result, _ = c2qa.util.simulate(self.circuit, shots=shots)

        # Convert qubit measurement results into base-10 Fock state counts
        fock_counts = c2qa.util.counts_to_fockcounts(
            self.circuit, result
        )
        print("Simulation complete.")
        return fock_counts

if __name__ == "__main__":
    # --- Define the Unbounded Knapsack Problem ---
    # Objective: Minimize -n1 - 2n2
    # Constraint: 8*n1 + 3*n2 <= 22  => 8*n1 + 3*n2 + n3 = 22
    # We need 3 qumodes: n1, n2 for items, and n3 for the slack variable.
    # The known optimal solution is n1=0, n2=7, which gives n3 = 22 - 3*7 = 1.
    # The expected final state should have the highest probability for |0, 7, 1>.
    NUM_QUMODES = 3

    # Cutoff needs to be high enough to represent the solution state (n2=7).
    # A cutoff of 8 (2^3) is sufficient.
    NUM_QUBITS_PER_QUMODE = 3

    # Penalty coefficient for the constraint
    LAMBDA = 0.5 # Penalty can be tuned for better performance

    # Constant from the constraint (8n1 + 3n2 + n3 - C)^2
    C = 22

    # --- Construct the Hamiltonian Term by Term ---
    # This matches the expansion of Hp = (-n1 - 2n2) + λ(8n1 + 3n2 + n3 - C)^2
    hamiltonian_specification = [
        # Objective terms
        {"type": "n", "qumode": 0, "coeff": -1.0},
        {"type": "n", "qumode": 1, "coeff": -2.0},

        # Penalty terms: Linear (from -2*C*λ*(...))
        {"type": "n", "qumode": 0, "coeff": LAMBDA * -2 * C * 8},
        {"type": "n", "qumode": 1, "coeff": LAMBDA * -2 * C * 3},
        {"type": "n", "qumode": 2, "coeff": LAMBDA * -2 * C * 1},

        # Penalty terms: Quadratic (λ * n_i^2)
        {"type": "n_squared", "qumode": 0, "coeff": LAMBDA * 8**2},
        {"type": "n_squared", "qumode": 1, "coeff": LAMBDA * 3**2},
        {"type": "n_squared", "qumode": 2, "coeff": LAMBDA * 1**2},

        # Penalty terms: Cross-Kerr (λ * 2 * n_i * n_j)
        {"type": "n1n2", "qumode": 0, "qumode2": 1, "coeff": LAMBDA * 2 * 8 * 3},
        {"type": "n1n2", "qumode": 0, "qumode2": 2, "coeff": LAMBDA * 2 * 8 * 1},
        {"type": "n1n2", "qumode": 1, "qumode2": 2, "coeff": LAMBDA * 2 * 3 * 1},
    ]

    # --- Instantiate and Run the Solver ---
    ukp_solver = AQC_MIP_Solver(
        num_qumodes=NUM_QUMODES,
        num_qubits_per_qumode=NUM_QUBITS_PER_QUMODE,
        hamiltonian_terms=hamiltonian_specification,
        total_time=150,
        num_trotter_steps=100,
        initial_squeezing=1.0,
    )

    # Solve the problem by simulating the circuit
    fock_counts = ukp_solver.solve(shots=4096)

    # --- Process and Display Results ---
    print("\n--- Simulation Results ---")
    print(f"Top {min(10, len(fock_counts))} measurement outcomes (Fock states |n1, n2, n3>):")

    # Sort results by counts in descending order
    sorted_counts = sorted(fock_counts.items(), key=lambda item: item[1], reverse=True)

    for i, (state, count) in enumerate(sorted_counts):
        if i >= 10:
            break
        print(f"  {state}: {count} counts")

    if sorted_counts:
        most_likely_state = sorted_counts[0][0]
        print(f"\nHighest probability state found: {most_likely_state}")
        print("This should be close to the expected optimal solution of (0, 7, 1).")