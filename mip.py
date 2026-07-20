import numpy as np
import qiskit
import c2qa
import warnings
from typing import List, Dict, Callable
import matplotlib.pyplot as plt

# 忽略scipy稀疏矩阵效率警告
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.sparse._index")

# 假设这些自定义门定义已存在于gate模块中
from gate import SQR, cv_R, cv_CRi_dag, add_RR

class AQC_MIP_Solver:
    """
    Solves Mixed-Integer Programming (MIP) problems using simulated
    Adiabatic Quantum Computation (AQC) with the c2qa library.
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
        """初始化AQC求解器"""
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
        self.cr = qiskit.ClassicalRegister(self.qmr.size, name="cr")
        self.circuit = c2qa.CVCircuit(self.qmr, self.ancqubit, self.cr)
        self.expectation_values = []  # 存储每一步的期望值
        # 预计算目标函数系数以避免硬编码
        self.objective_coeffs = self._extract_objective_coefficients()
        # 计算总 qubit 数量（辅助 qubit + 每个 qumode 的 qubit）
        self.total_qubits = self.ancqubit.size + self.qmr.size

    def _extract_objective_coefficients(self):
        """从哈密顿量中提取目标函数系数"""
        coeffs = [0.0] * self.num_qumodes
        for term in self.hamiltonian_terms:
            if term.get("type") == "n" and term.get("coeff") < 0:  # 假设负系数为目标项
                qumode_idx = term.get("qumode")
                coeffs[qumode_idx] = term.get("coeff")
        return coeffs
    def _apply_hamiltonian_gates(self, hamiltonian_schedule: float):
        """应用问题哈密顿量对应的门"""
        for term in self.hamiltonian_terms:
            op_type = term.get("type")
            qumode_idx = term.get("qumode")
            qumode_idx_2 = term.get("qumode2")
            coeff = term.get("coeff")
            angle = coeff * hamiltonian_schedule * self.dt

            if op_type == "n":
                self.circuit.cv_r(-angle, self.qmr[qumode_idx])
            
            elif op_type == "n_squared":
                phases = [-angle * (k**2) for k in range(self.fock_cutoff)]
                self.circuit.cv_snap(phases, list(range(self.fock_cutoff)), self.qmr[qumode_idx])

            elif op_type == "n1n2":
                add_RR(self.circuit, self.ancqubit, self.qmr[qumode_idx], self.qmr[qumode_idx_2], 2*angle, self.fock_cutoff)

            else:
                print(f"Warning: Hamiltonian term type '{op_type}' is not recognized.")

    def _apply_mixing_hamiltonian_gates(self, mixer_schedule: float):
        """应用混合哈密顿量对应的门"""
        angle = mixer_schedule * self.dt
        for i in range(self.num_qumodes):
            self.circuit.cv_sq(angle, self.qmr[i])
    def _calculate_expectation(self, statevector):
        """使用stateread()计算目标函数的期望值"""
        expectation = 0.0
            # 使用stateread解析状态向量
            # 参数: 状态向量, 量子比特数, 量子模数, 截断值
        _, states = c2qa.util.stateread(
                statevector, 
                numberofqubits=self.total_qubits,
                numberofmodes=self.num_qumodes,
                cutoff=self.fock_cutoff,
                verbose=False,
                little_endian=False
        )
        print(states)
        # 遍历所有状态计算期望值
        for state in states:
                fock_nums, _, amplitude = state
                prob = abs(amplitude)** 2
                # 确保Fock数有效
                valid = True
                for n in fock_nums:
                    if n >= self.fock_cutoff:
                        valid = False
                        break
                
                if valid:
                    # 计算目标值
                    value = sum(c * n for c, n in zip(self.objective_coeffs, fock_nums))
                    expectation += value * prob
            
        return expectation

    def plot_expectation_values(self):
        """绘制演化过程中的期望值变化"""
        if not self.expectation_values:
            print("没有记录的期望值，请在build_circuit时设置measure_intermediate=True")
            return
            
        steps = [item['step'] for item in self.expectation_values]
        expectations = [item['expectation'] for item in self.expectation_values]
        schedules = [item['schedule'] for item in self.expectation_values]
        
        # 创建双轴图，同时显示步数和调度参数
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Trotter Step')
        ax1.set_ylabel('Objective Expectation', color=color)
        ax1.plot(steps, expectations, 'b-', linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Adiabatic Schedule s(t)', color=color)
        ax2.plot(steps, schedules, 'r--', alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Adiabatic Evolution: Objective Expectation vs. Step')
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()

    def build_circuit(self, measure_intermediate=True):
        """构建完整的AQC电路，新增中间态测量选项"""
        print("Building AQC circuit...")
        # 1. 准备初始态
        for i in range(self.num_qumodes):
            self.circuit.cv_sq(self.initial_squeezing, self.qmr[i])

        self.circuit.barrier()

        # 2. 绝热演化（Trotter分解）
        for step in range(self.num_trotter_steps):
            schedule = (step + 1) / self.num_trotter_steps

            # 应用问题哈密顿量门
            self._apply_hamiltonian_gates(hamiltonian_schedule=schedule)

            # 应用混合哈密顿量门
            self._apply_mixing_hamiltonian_gates(mixer_schedule=(1.0 - schedule))

            # 计算并记录中间期望值
            if measure_intermediate:
                try:
                    # 使用c2qa的simulate函数获取状态向量（不添加测量）
                    temp_circuit = self.circuit.copy()
                    _, result, _ = c2qa.util.simulate(temp_circuit, shots=1)
                    
                    # 从结果中提取状态向量
                    statevector = result.get_statevector()
                    
                    # 计算期望值
                    expectation = self._calculate_expectation(statevector)
                    self.expectation_values.append({
                        'step': step,
                        'schedule': schedule,
                        'expectation': expectation
                    })
                except Exception as e:
                    print(f"步骤 {step} 计算期望值失败: {str(e)}")

            self.circuit.barrier()

        # 3. 添加最终测量
        self.circuit.cv_measure(self.qmr, self.cr)
        print("Circuit build complete.")


    def solve(self, shots: int = 2048):
        """模拟AQC电路并返回结果"""
        if not self.circuit.data:
            self.build_circuit()

        print(f"Simulating circuit with {shots} shots...")
        _, result, _ = c2qa.util.simulate(self.circuit, shots=shots)

        # 转换测量结果为Fock态计数
        fock_counts = c2qa.util.counts_to_fockcounts(
            self.circuit, result
        )
        print("Simulation complete.")
        return fock_counts

if __name__ == "__main__":
    # 定义无界背包问题
    NUM_QUMODES = 3
    NUM_QUBITS_PER_QUMODE = 3  # 8的截断足够表示解
    LAMBDA = 0.7  # 惩罚系数
    C = 6  # 约束常数

    # 构建哈密顿量项
    hamiltonian_specification = [
        # 目标项
        {"type": "n", "qumode": 0, "coeff": -1.0},
        {"type": "n", "qumode": 1, "coeff": -2.0},
        {"type": "n", "qumode": 2, "coeff": -1.0},

        # 惩罚项：线性部分
        {"type": "n", "qumode": 0, "coeff": LAMBDA * -2 * C * 3},
        {"type": "n", "qumode": 1, "coeff": LAMBDA * -2 * C * 1},
        {"type": "n", "qumode": 2, "coeff": LAMBDA * -2 * C * 1},

        # 惩罚项：二次项
        {"type": "n_squared", "qumode": 0, "coeff": LAMBDA * 3**2},
        {"type": "n_squared", "qumode": 1, "coeff": LAMBDA * 1**2},
        {"type": "n_squared", "qumode": 2, "coeff": LAMBDA * 1**2},

        # 惩罚项：交叉项
        {"type": "n1n2", "qumode": 0, "qumode2": 1, "coeff": LAMBDA * 2 * 3 * 1},
        {"type": "n1n2", "qumode": 0, "qumode2": 2, "coeff": LAMBDA * 2 * 3 * 1},
        {"type": "n1n2", "qumode": 1, "qumode2": 2, "coeff": LAMBDA * 2 * 1 * 1},
    ]

    # 实例化求解器
    ukp_solver = AQC_MIP_Solver(
        num_qumodes=NUM_QUMODES,
        num_qubits_per_qumode=NUM_QUBITS_PER_QUMODE,
        hamiltonian_terms=hamiltonian_specification,
        total_time=300,
        num_trotter_steps=200,
        initial_squeezing=1.0,
    )

    # 构建电路并启用中间期望值测量
    ukp_solver.build_circuit(measure_intermediate=True)
    # 求解问题
    fock_counts = ukp_solver.solve(shots=4096)

    # 绘制期望值曲线
    ukp_solver.plot_expectation_values()

    # 处理并显示结果
    print("\n--- Simulation Results ---")
    print(f"Top {min(10, len(fock_counts))} measurement outcomes (Fock states |n1, n2, n3>):")

    sorted_counts = sorted(fock_counts.items(), key=lambda item: item[1], reverse=True)

    for i, (state, count) in enumerate(sorted_counts):
        if i >= 10:
            break
        print(f"  {state}: {count} counts")

    if sorted_counts:
        most_likely_state = sorted_counts[0][0]
        print(f"\nHighest probability state found: {most_likely_state}")
        print("This should be close to the expected optimal solution of (0, 6, 0).")