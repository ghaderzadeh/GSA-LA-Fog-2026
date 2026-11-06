# Context-Aware Application Distribution in Fog Environments (GSA-LA)

Official anonymized implementation for the manuscript:
**"Context-Aware Application Distribution in Fog Environments: Enhancing QoS Using a Hybrid Gravitational Search and Machine Learning Algorithm"**

## Overview
This repository contains the Python simulation environment, topology generators, workload scale-down mappers, and optimization engines proposed in the paper[cite: 4, 5]:
- **19-Dimensional Context Vector ($\mathbf{S}_j(t)$):** Real-time monitoring across CPU, RAM, Swap, Disk I/O, Network Traffic, and Graph-derived Propagation Latencies[cite: 2, 4].
- **Barabási–Albert Scale-Free Topology:** Shortest-path hop latency and bandwidth derivation ($J \in \{20, 40, \dots, 180\}$ nodes)[cite: 4, 5].
- **Alibaba Cluster Trace Scale-Down Mapper:** Statistical scaling ($10^{-2}$ to $10^{-3}$) to emulate transient IoT microservices with bursty Poisson arrivals ($\lambda \in [10, 50]$ req/s).
- **Hybrid GSA-LA Engine (Algorithm 1):** Gravitational Search Algorithm coupled with a state-quantized Contextual Bandit Learning Automaton ($L_{R-P}$ scheme), Heuristic Feasibility Repair Operator ($\sum_j x_{ij} = 1$), and $M/M/c_j$ queuing delay estimation[cite: 2, 4].
- **Baselines & Statistical Suite:** Implementations of BPSO, ACO, GWO, Standard BGSA, Pure-LA, HAGSA-NS, LinUCB, and Budgeted-DQN, along with Friedman, Nemenyi, Holm-Bonferroni Wilcoxon tests, and Cliff's delta ($\delta$).

## Installation & Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt