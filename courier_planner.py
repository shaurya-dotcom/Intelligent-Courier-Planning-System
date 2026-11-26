"""
Intelligent Courier Planning System - DAA Capstone Project
Integrates:
  - Phase 1: Problem Design & Data Modeling
  - Phase 2: Merge Sort (Prioritization) & 0/1 Knapsack DP (Optimal Box Packing)
  - Phase 3: Dijkstra's Algorithm (Shortest Paths) & Nearest-Neighbor TSP Approximation (Route Planning)
  - Phase 4: Scalability Benchmarking & Runtime Performance Visualization
"""

import csv
import json
import heapq
import time
import random
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# PHASE 1: DATA STRUCTURES & GRAPH MODELING
# ==============================================================================
class Parcel:
    def __init__(self, parcel_id, destination, weight, priority_value, deadline_hr):
        self.parcel_id = parcel_id
        self.destination = destination
        self.weight = int(weight)
        self.priority_value = int(priority_value)
        self.deadline_hr = int(deadline_hr)

    def __repr__(self):
        return (f"Parcel({self.parcel_id}, dest={self.destination}, "
                f"wt={self.weight}kg, val={self.priority_value}, dl={self.deadline_hr}h)")


class DeliveryGraph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, distance):
        self.adj.setdefault(u, []).append((v, float(distance)))
        self.adj.setdefault(v, []).append((u, float(distance)))

    def nodes(self):
        return list(self.adj.keys())


def load_parcels(filepath):
    parcels = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parcels.append(Parcel(
                row["parcel_id"],
                row["destination"],
                row["weight"],
                row["priority_value"],
                row["deadline_hr"]
            ))
    return parcels


def load_graph(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    g = DeliveryGraph()
    for u, v, w in data["edges"]:
        g.add_edge(u, v, w)
    return data["depot"], g


# ==============================================================================
# PHASE 2: SORTING (MERGE SORT) & BOX PACKING (0/1 KNAPSACK DP)
# ==============================================================================
def merge_sort_parcels(parcels):
    """
    Sorts parcels by earliest deadline_hr ascending, then highest priority_value descending.
    Time Complexity: O(n log n) guaranteed in best, average, and worst cases.
    """
    if len(parcels) <= 1:
        return parcels

    mid = len(parcels) // 2
    left = merge_sort_parcels(parcels[:mid])
    right = merge_sort_parcels(parcels[mid:])
    return _merge(left, right)


def _merge(left, right):
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        # Primary key: earlier deadline; Secondary key: higher priority value
        if (left[i].deadline_hr, -left[i].priority_value) <= (right[j].deadline_hr, -right[j].priority_value):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def knapsack_01_dp(parcels, capacity):
    """
    0/1 Knapsack Dynamic Programming to select optimal parcels within vehicle weight capacity.
    Time Complexity: O(n * W), Space Complexity: O(n * W)
    Returns: (max_value, selected_parcels, dp_table)
    """
    n = len(parcels)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        wt = parcels[i - 1].weight
        val = parcels[i - 1].priority_value
        for w in range(capacity + 1):
            if wt <= w:
                dp[i][w] = max(dp[i - 1][w], val + dp[i - 1][w - wt])
            else:
                dp[i][w] = dp[i - 1][w]

    # Backtrack to find selected parcels
    selected = []
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(parcels[i - 1])
            w -= parcels[i - 1].weight

    selected.reverse()
    return dp[n][capacity], selected, dp


# ==============================================================================
# PHASE 3: GRAPH SHORTEST PATHS (DIJKSTRA) & ROUTE APPROXIMATION (NN-TSP)
# ==============================================================================
def dijkstra(graph, start):
    """
    Computes shortest paths from start node to all reachable nodes using a min-heap.
    Time Complexity: O((V + E) log V)
    """
    distances = {node: float('inf') for node in graph.nodes()}
    previous = {node: None for node in graph.nodes()}
    distances[start] = 0.0
    pq = [(0.0, start)]

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if curr_dist > distances[u]:
            continue
        for v, weight in graph.adj.get(u, []):
            alt = curr_dist + weight
            if alt < distances[v]:
                distances[v] = alt
                previous[v] = u
                heapq.heappush(pq, (alt, v))

    return distances, previous


def reconstruct_path(previous, start, target):
    path = []
    curr = target
    while curr is not None:
        path.append(curr)
        if curr == start:
            break
        curr = previous[curr]
    path.reverse()
    return path if path and path[0] == start else []


def nearest_neighbor_tsp(graph, depot, stops):
    """
    Approximates the Travelling Salesperson Problem (TSP) route starting and ending at depot,
    visiting all unique delivery stops using Dijkstra shortest-path distances.
    Time Complexity: O(k^2 + k * (V + E) log V) where k = number of unique stops.
    """
    unique_stops = list(dict.fromkeys(stops))
    if depot in unique_stops:
        unique_stops.remove(depot)

    all_points = [depot] + unique_stops
    dist_matrix = {}
    path_matrix = {}

    for pt in all_points:
        dists, prev = dijkstra(graph, pt)
        dist_matrix[pt] = dists
        path_matrix[pt] = prev

    unvisited = set(unique_stops)
    current = depot
    tour_stops = [depot]
    full_road_segments = []
    total_distance = 0.0

    while unvisited:
        nearest = min(unvisited, key=lambda city: dist_matrix[current][city])
        leg_dist = dist_matrix[current][nearest]
        leg_path = reconstruct_path(path_matrix[current], current, nearest)
        total_distance += leg_dist
        full_road_segments.append((current, nearest, round(leg_dist, 2), leg_path))
        tour_stops.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    # Return to depot
    return_dist = dist_matrix[current][depot]
    return_path = reconstruct_path(path_matrix[current], current, depot)
    total_distance += return_dist
    full_road_segments.append((current, depot, round(return_dist, 2), return_path))
    tour_stops.append(depot)

    return tour_stops, round(total_distance, 2), full_road_segments


# ==============================================================================
# PHASE 4: PERFORMANCE TESTING & RUNTIME BENCHMARKING
# ==============================================================================
def run_benchmarks(output_chart_path):
    sizes = [50, 150, 300, 600, 1000]
    sort_times_ms = []
    dp_times_ms = []
    tsp_times_ms = []

    random.seed(42)
    capacity = 100

    for n in sizes:
        # Generate synthetic parcels
        syn_parcels = [
            Parcel(f"P{i}", f"Node_{i % 25}", random.randint(1, 10), random.randint(10, 100), random.randint(1, 12))
            for i in range(n)
        ]

        # Generate synthetic connected graph with min(n, 40) nodes
        num_nodes = min(max(10, n // 10), 50)
        g = DeliveryGraph()
        nodes = ["Depot"] + [f"Node_{i}" for i in range(num_nodes)]
        for i in range(len(nodes) - 1):
            g.add_edge(nodes[i], nodes[i + 1], round(random.uniform(1.0, 10.0), 2))
        for _ in range(num_nodes * 2):
            u, v = random.sample(nodes, 2)
            g.add_edge(u, v, round(random.uniform(1.5, 12.0), 2))

        # 1. Measure Merge Sort
        t0 = time.perf_counter()
        sorted_p = merge_sort_parcels(syn_parcels)
        t1 = time.perf_counter()
        sort_times_ms.append((t1 - t0) * 1000)

        # 2. Measure 0/1 Knapsack DP
        t0 = time.perf_counter()
        knapsack_01_dp(sorted_p, capacity)
        t1 = time.perf_counter()
        dp_times_ms.append((t1 - t0) * 1000)

        # 3. Measure Dijkstra + Nearest-Neighbor TSP
        stops = [f"Node_{i}" for i in range(min(num_nodes, 20))]
        t0 = time.perf_counter()
        nearest_neighbor_tsp(g, "Depot", stops)
        t1 = time.perf_counter()
        tsp_times_ms.append((t1 - t0) * 1000)

    # Plot runtime comparison chart
    plt.figure(figsize=(9, 5))
    plt.plot(sizes, sort_times_ms, marker='o', linewidth=2, label='Merge Sort O(n log n)')
    plt.plot(sizes, dp_times_ms, marker='s', linewidth=2, label='0/1 Knapsack DP O(n * W)')
    plt.plot(sizes, tsp_times_ms, marker='^', linewidth=2, label='Dijkstra + NN-TSP O(k^2 + k(V+E)log V)')
    plt.title('Algorithmic Runtime Performance Across Input Sizes (n)', fontsize=13, fontweight='bold')
    plt.xlabel('Number of Parcels (n)', fontsize=11)
    plt.ylabel('Execution Time (milliseconds)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=200)
    plt.close()

    return list(zip(sizes, sort_times_ms, dp_times_ms, tsp_times_ms))


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parcels_file = os.path.join(base_dir, "datasets", "parcels_small.csv")
    graph_file = os.path.join(base_dir, "datasets", "city_graph.json")
    chart_file = os.path.join(base_dir, "outputs", "runtime_performance.png")
    trace_file = os.path.join(base_dir, "outputs", "execution_trace.txt")

    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("INTELLIGENT COURIER PLANNING SYSTEM - EXECUTION TRACE")
    log("=" * 72)

    # Phase 1: Load Data
    parcels = load_parcels(parcels_file)
    depot, graph = load_graph(graph_file)
    vehicle_capacity = 15
    log("")
    log(f"[PHASE 1] Loaded {len(parcels)} parcels | Depot: {depot} | Vehicle Capacity: {vehicle_capacity} kg")
    log(f"Graph Nodes ({len(graph.nodes())}): {', '.join(graph.nodes())}")

    # Phase 2A: Merge Sort
    sorted_parcels = merge_sort_parcels(parcels)
    log("")
    log("[PHASE 2A] Parcels Sorted by Deadline (Asc) & Priority Value (Desc) via Merge Sort:")
    for p in sorted_parcels:
        log(f"  - {p}")

    # Phase 2B: 0/1 Knapsack DP
    max_val, packed_parcels, dp_table = knapsack_01_dp(sorted_parcels, vehicle_capacity)
    total_wt = sum(p.weight for p in packed_parcels)
    log("")
    log(f"[PHASE 2B] 0/1 Knapsack DP Optimal Packing Plan (Capacity = {vehicle_capacity} kg):")
    log(f"  - Total Priority Value Achieved : {max_val}")
    log(f"  - Total Weight Loaded           : {total_wt} / {vehicle_capacity} kg")
    log("  - Selected Parcels:")
    for p in packed_parcels:
        log(f"      * {p.parcel_id} -> {p.destination} (Weight: {p.weight} kg, Value: {p.priority_value}, Deadline: {p.deadline_hr}h)")

    log("")
    log("  - DP Table Snapshot (Rows: Items 0..8, Cols: Capacity 0..15):")
    header = "    Item\\Cap | " + " ".join(f"{w:>3}" for w in range(vehicle_capacity + 1))
    log(header)
    log("    " + "-" * (len(header) - 4))
    for i_row, row in enumerate(dp_table):
        label = "Init(0)" if i_row == 0 else sorted_parcels[i_row - 1].parcel_id
        log(f"    {label:<8} | " + " ".join(f"{val:>3}" for val in row))

    # Phase 3: Dijkstra & Nearest-Neighbor TSP
    delivery_stops = [p.destination for p in packed_parcels]
    tour_stops, total_dist, legs = nearest_neighbor_tsp(graph, depot, delivery_stops)
    log("")
    log("[PHASE 3] Route Optimization (Dijkstra Shortest Paths + Nearest-Neighbor TSP):")
    log(f"  - Delivery Stop Sequence : {' -> '.join(tour_stops)}")
    log(f"  - Total Route Distance   : {total_dist} km")
    log("  - Detailed Road Legs (via Dijkstra):")
    for u, v, d, path in legs:
        log(f"      * {u} -> {v} ({d} km) | Road Path: {' -> '.join(path)}")

    # Phase 4: Performance Benchmarking
    bench_results = run_benchmarks(chart_file)
    log("")
    log("[PHASE 4] Runtime Scalability Benchmark (in milliseconds):")
    log(f"  {'Input (n)':<12} | {'Merge Sort (ms)':<16} | {'0/1 Knapsack DP (ms)':<21} | {'Dijkstra + TSP (ms)':<20}")
    log("  " + "-" * 76)
    for n, t_sort, t_dp, t_tsp in bench_results:
        log(f"  {n:<12} | {t_sort:<16.4f} | {t_dp:<21.4f} | {t_tsp:<20.4f}")
    log("")
    log(f"  [Saved runtime chart to: {chart_file}]")

    with open(trace_file, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
