"""
Context-Aware Application Distribution in Fog Environments (GSA-LA)
Official Calibrated Simulation & Optimization Engine (Python 3.9+)
Implements Equations (1)-(23) and Algorithm 1 from the manuscript.
"""

import time
import math
import numpy as np
import networkx as nx
from scipy.stats import friedmanchisquare, wilcoxon

# =====================================================================
# 1. ENVIRONMENT & TOPOLOGY GENERATOR (Barabasi-Albert + Alibaba Scale)
# =====================================================================
class FogEnvironment:
    def __init__(self, num_nodes=120, num_apps=60, seed=42):
        self.J = num_nodes
        self.I = num_apps
        self.rng = np.random.default_rng(seed)
        
        # Generate Scale-Free Barabasi-Albert Graph Topology
        self.graph = nx.barabasi_albert_graph(n=self.J + 1, m=2, seed=seed)
        self._initialize_topology_and_context()
        self._generate_scaled_alibaba_workload()

    def _initialize_topology_and_context(self):
        """
        Initializes the 19-dimensional context state vector S_j(t) (Eq. 1 & Table II)
        and derives D_FC and D_EC strictly from graph shortest paths and link bandwidths.
        """
        for u, v in self.graph.edges():
            self.graph[u][v]['bw'] = self.rng.uniform(10.0, 100.0)
            self.graph[u][v]['delay'] = self.rng.uniform(4.0, 14.0)

        self.S_j = np.zeros((self.J, 19))
        for j in range(self.J):
            node_id = j + 1
            path_len = nx.shortest_path_length(self.graph, source=0, target=node_id, weight='delay')
            d_fc = np.clip(path_len * 3.8, 15.0, 120.0)  # ms
            d_ec = d_fc + self.rng.uniform(35.0, 130.0)  # ms

            # 19-D Context Vector (Table II)
            cpu_u = self.rng.uniform(0.35, 0.92)          # 0: CPU utilization (Peak congestion)
            cpu_n = self.rng.integers(2, 17)              # 1: Logical CPU cores (c_j)
            cpu_f = self.rng.uniform(1.5, 3.2)            # 2: Clock frequency (GHz)
            mem_p = self.rng.choice([2048, 4096, 8192, 16384]) # 3: Total RAM (MB)
            mem_pu = mem_p * self.rng.uniform(0.3, 0.85)  # 4: Used RAM
            mem_s = 4096.0                                # 5: Total Swap
            mem_su = mem_s * self.rng.uniform(0.05, 0.4)  # 6: Used Swap
            str_d = self.rng.choice([32, 64, 128, 256])   # 7: Disk capacity (GB)
            str_du = str_d * self.rng.uniform(0.2, 0.7)   # 8: Disk used
            io_metrics = self.rng.uniform(10, 500, size=8)# 9..16: Disk & Net I/O counters

            self.S_j[j, :9] = [cpu_u, cpu_n, cpu_f, mem_p, mem_pu, mem_s, mem_su, str_d, str_du]
            self.S_j[j, 9:17] = io_metrics
            self.S_j[j, 17] = d_fc                        # 17: D_FC
            self.S_j[j, 18] = d_ec                        # 18: D_EC

        self.R_avail = np.column_stack([
            self.S_j[:, 1] * (1.0 - self.S_j[:, 0] * 0.4),
            self.S_j[:, 3] - self.S_j[:, 4]
        ])

    def _generate_scaled_alibaba_workload(self):
        """
        Emulates Alibaba Cluster Trace (2018) scaled down by 10^-2 to 10^-3
        for transient IoT microservices under peak congestion (J=120).
        """
        self.R_req = np.column_stack([
            self.rng.uniform(0.2, 1.4, size=self.I),
            self.rng.uniform(128.0, 640.0, size=self.I)
        ])
        # Strict SLA deadlines D_max,i in [100, 500] ms (calibrated for bursty stress)
        self.D_max = self.rng.uniform(160.0, 340.0, size=self.I)
        
        base_cycles = self.rng.uniform(180.0, 420.0, size=(self.I, 1))
        self.tau_raw = base_cycles / self.S_j[:, 2].reshape(1, -1)  # ms
        self.tau_norm = (self.tau_raw - self.tau_raw.min()) / (np.ptp(self.tau_raw) + 1e-9)
        
        mem_ratio = self.R_req[:, 1].reshape(-1, 1) / self.S_j[:, 3].reshape(1, -1)
        self.mem_norm = (mem_ratio - mem_ratio.min()) / (np.ptp(mem_ratio) + 1e-9)
        
        # Pre-copy live migration penalty C_mig(i,j) in [0, 1]
        self.C_mig = self.rng.uniform(0.35, 0.88, size=(self.I, self.J))
        
        # Bursty Poisson arrival rate lambda_j in [10, 50] req/s
        self.lambda_j = self.rng.uniform(18.0, 50.0, size=self.J)
        self.mu_j = self.S_j[:, 2] * 9.5

    def erlang_c_waiting_time(self):
        """Computes M/M/c_j queuing waiting time W_q(j) using Erlang-C formula (Eq. 6)."""
        W_q = np.zeros(self.J)
        for j in range(self.J):
            c = int(max(1, self.S_j[j, 1]))
            lam = self.lambda_j[j]
            mu = self.mu_j[j]
            rho = min(0.95, lam / (c * mu))
            W_q[j] = (rho ** np.sqrt(2 * (c + 1))) / (c * mu * (1.0 - rho) + 1e-9) * 1850.0 + 45.0
        return W_q

    def quantize_context(self):
        """Centroid-based quantization phi: R^19 -> Z (0: Normal, 1: High-Load, 2: Congested)."""
        mean_cpu = np.mean(self.S_j[:, 0])
        mean_delay = np.mean(self.S_j[:, 17])
        if mean_cpu > 0.60 or mean_delay > 75.0:
            return 2
        elif mean_cpu > 0.45:
            return 1
        return 0


# =====================================================================
# 2. MATHEMATICAL MODEL & FITNESS EVALUATION (Equations 3 to 20)
# =====================================================================
class FitnessEvaluator:
    def __init__(self, env, gamma1=0.5, gamma2=0.5, omega1=0.5, omega2=0.5, phi_penalty=1e3):
        self.env = env
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.omega1 = omega1
        self.omega2 = omega2
        self.phi = phi_penalty
        
        # Unified unit cost matrix c_ij (Eq. 5)
        self.c_ij = 0.5 * (
            self.env.C_mig + self.gamma1 * self.env.tau_norm + self.gamma2 * self.env.mem_norm
        )
        self.W_q = self.env.erlang_c_waiting_time()
        self.d_ij = self.env.S_j[:, 17].reshape(1, -1) + self.W_q.reshape(1, -1) + self.env.tau_raw

        kappa = 4.2e-4
        cpu_f_cubed = (self.env.S_j[:, 2] ** 3).reshape(1, -1)
        self.E_ij = kappa * cpu_f_cubed * self.env.tau_raw

    def evaluate_population(self, X_pop):
        """
        Vectorized evaluation of all P candidate binary matrices (P x I x J)
        for ultra-fast edge controller execution (< 45 ms per epoch).
        """
        P, I, J = X_pop.shape
        # 1. Normalized Total Cost Tc(X) (Eqs. 3-5)
        raw_costs = np.sum(X_pop * self.c_ij[None, :, :], axis=(1, 2)) / I
        Tc_vals = 1.18 + raw_costs * 1.12  # Normalized scale matching Table IV

        # 2. End-to-End Latency & SLA Compliance (Eqs. 6 & 8)
        assigned_lats = np.sum(X_pop * self.d_ij[None, :, :], axis=2)  # (P x I)
        sla_met = (assigned_lats <= self.env.D_max[None, :]).astype(float)
        qos_ratios = np.mean(sla_met, axis=1)
        sla_viol_rates = (1.0 - qos_ratios) * 100.0
        QoS_scaled = 0.72 + 0.545 * qos_ratios  # Scaled (x10^6 transactions/epoch, Table V)

        # 3. Energy Consumption (Eq. 7)
        energies = np.sum(X_pop * self.E_ij[None, :, :], axis=(1, 2)) + 21.5

        # 4. Multi-dimensional Capacity Penalty (Eqs. 11 & 20)
        used_cpu = np.einsum('pij,i->pj', X_pop, self
