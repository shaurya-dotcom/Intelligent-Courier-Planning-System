# Intelligent Courier Planning System — DAA Capstone Project

## Project Overview
An integrated Python prototype that models and solves a multi-stage logistics and courier delivery problem by combining core **Design and Analysis of Algorithms (DAA)** paradigms:
1. **Divide & Conquer (Merge Sort)** for deadline- and priority-based parcel ordering.
2. **Dynamic Programming (0/1 Knapsack)** for optimal vehicle cargo packing under strict weight capacity.
3. **Graph Algorithms (Dijkstra's Shortest Path)** & **Approximation Algorithms (Nearest-Neighbor TSP)** for minimum-distance delivery routing.
4. **Empirical Performance Benchmarking** across scaling input sizes ($n = 50$ to $n = 1000$).

---

## Project Structure
```text
Courier_Planning_Capstone/
├── courier_planner.py              # Complete integrated Python implementation
├── README.md                       # Full documentation, design analysis, and execution guide
├── datasets/
│   ├── parcels_small.csv           # Sample dataset of 8 parcels with weights, values & deadlines
│   └── city_graph.json             # Weighted road network graph of delivery hubs & sectors
└── outputs/
    ├── execution_trace.txt         # Full console execution output & DP table trace
    └── runtime_performance.png     # Runtime scalability comparison chart
```

---

## Phase-by-Phase Technical Design

### Phase 1: Courier Problem Design and Planning
- **Entities:**
  - `Parcel(parcel_id, destination, weight, priority_value, deadline_hr)`
  - `DeliveryGraph`: Undirected weighted adjacency list representing road intersections/sectors and distances (km).
- **Pipeline Architecture:**
  1. Ingest parcel manifests and road network topology.
  2. Sort parcels by urgency (`deadline_hr` ascending, `priority_value` descending).
  3. Pack vehicle up to capacity $W$ maximizing total `priority_value`.
  4. Extract unique destinations of packed parcels, compute all-pairs shortest road distances among those stops via Dijkstra, and construct a closed delivery tour starting/ending at `Hub_Central` using Nearest-Neighbor TSP.

### Phase 2: Package Prioritization & Box Filling
- **Sorting Algorithm — Merge Sort ($O(n \log n)$):**
  - Chosen over Quick Sort because Merge Sort guarantees $O(n \log n)$ worst-case performance and stability, preserving relative order among equal-deadline parcels.
- **Box Packing — 0/1 Knapsack Dynamic Programming ($O(n \cdot W)$):**
  - Parcels cannot be split (0/1 constraint), making greedy fractional knapsack invalid.
  - Recurrence:
    $$DP[i][w] = \max(DP[i-1][w],\; v_i + DP[i-1][w - w_i]) \quad \text{for } w_i \le w$$
  - Backtracking through the DP table recovers the exact optimal parcel subset.

### Phase 3: Route Maker & Graph Processing
- **Shortest Path — Dijkstra's Algorithm ($O((V + E) \log V)$):**
  - Uses a binary min-heap (`heapq`) to compute exact shortest road distances and parent pointers between the depot and every delivery destination.
- **Tour Construction — Nearest-Neighbor TSP Approximation ($O(k^2)$ over $k$ stops):**
  - Exact TSP is NP-Hard ($O(k!)$). Nearest-Neighbor greedily visits the closest unvisited stop at each step and returns to the depot, providing fast polynomial-time routing.

### Phase 4: Performance Testing & Future Optimization
- Benchmarked across $n \in \{50, 150, 300, 600, 1000\}$ parcels (`outputs/runtime_performance.png`).
- **Observed Scaling:**
  - Merge Sort scales smoothly as $O(n \log n)$.
  - 0/1 Knapsack scales linearly with $n$ for fixed capacity $W$ ($O(nW)$).
  - Dijkstra + NN-TSP depends primarily on graph size $(V, E)$ and unique stops $k$.
- **Future Optimizations:**
  1. Space-optimize 0/1 Knapsack to $O(W)$ 1D array when full DP table tracing is unnecessary.
  2. Upgrade Nearest-Neighbor TSP with **2-opt local search** to eliminate crossing tour edges and improve tour length by 10–15%.
  3. Add time-window constraints (VRPTW) directly into the routing heuristic.

---

## How to Run
```bash
# Ensure matplotlib is installed
pip install matplotlib

# Run the complete pipeline
python courier_planner.py
```
