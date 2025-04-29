import argparse
import os
from pathlib import Path
import os # Add this if os is not already imported
import gurobipy as gp
import networkx as nx
from gurobipy import GRB




def read_instance_file(filename: str | os.PathLike) -> nx.Graph:
    # This function reads the data file and creates a NetworkX graph.
    # Assumes the file format described in the comments within the function.
    with open(Path(filename), mode="r", encoding="utf-8") as f:
        n_nodes = int(f.readline())
        n_edges = int(f.readline())

        graph = nx.Graph()

        # skip comment line
        f.readline()

        # read node lines
        for i in range(n_nodes):
            line = f.readline()
            # Assumes format: node_id name supply_demand (e.g., "1 A 12")
            # Node IDs in the file are assumed to be 1 to n_nodes
            node_id_str, name, supply_demand = line.split()
            node_id = i + 1 # Use 1-based indexing consistent with typical file formats
            graph.add_node(node_id, name=name, supply_demand=int(supply_demand))

        # skip comment line
        f.readline()

        # read edge lines
        for _ in range(n_edges):
            line = f.readline()
            # Assumes format: edge_id node_1 node_2 transport_cost build_cost_1 build_cost_2 capacity_1 capacity_2
            (
                edge_id,
                node_1,
                node_2,
                transport_cost,
                build_cost_1,
                build_cost_2,
                capacity_1,
                capacity_2,
            ) = line.split()
            # Add edge data using integer node ids (assuming they are 1-based in the file)
            u = int(node_1)
            v = int(node_2)
            # Ensure we store edge data consistently, e.g., always with smaller node first if needed
            # Or just use the (u,v) pair as given if graph.edges handles undirected nature.
            graph.add_edge(
                u,
                v,
                id=int(edge_id),
                transport_cost=int(transport_cost),
                build_cost_1=int(build_cost_1),
                build_cost_2=int(build_cost_2),
                capacity_1=int(capacity_1),
                capacity_2=int(capacity_2),
            )

        return graph


def build_model(model: gp.Model, graph: nx.Graph):
    # --- Decision variables ---
    # Use graph.edges which provides undirected pairs (u, v)
    # Ensure consistent indexing if needed, e.g., using (min(u,v), max(u,v))
    # For simplicity, we use the pairs as given by graph.edges
    y1 = model.addVars(graph.edges, vtype=GRB.BINARY, name="y1")
    y2 = model.addVars(graph.edges, vtype=GRB.BINARY, name="y2")

    # Flow variables using a dictionary like the original script
    f = {}
    # graph.nodes gives the node IDs (e.g., 1, 2, ..., n_nodes)
    nodes = list(graph.nodes)
    for i in nodes:
        for j in graph.neighbors(i):
             # Define flow variable only if it doesn't exist to avoid duplicates for (j,i)
             if (i,j) not in f:
                  f[(i, j)] = model.addVar(lb=0.0, name=f"f_{i}_{j}")
             if (j,i) not in f:
                  f[(j, i)] = model.addVar(lb=0.0, name=f"f_{j}_{i}")


    model.update()

    # --- Objective ---
    build_cost = gp.quicksum(
        graph.edges[i, j]['build_cost_1'] * y1[i, j] +
        graph.edges[i, j]['build_cost_2'] * y2[i, j]
        for i, j in graph.edges # Iterate through unique undirected edges
    )

    # Transport cost: Sum c_ij * (f_ij + f_ji) for each edge (i,j)
    trans_cost = gp.quicksum(
        graph.edges[i, j]['transport_cost'] * (f.get((i, j), 0) + f.get((j, i), 0))
        for i, j in graph.edges # Iterate through unique undirected edges
    )

    model.setObjective(build_cost + trans_cost, GRB.MINIMIZE)

    # --- Constraints ---
    # (1) At most one build option per edge
    model.addConstrs(
        (y1[i, j] + y2[i, j] <= 1 for i, j in graph.edges),
        name="onebuild"
    )

    # (2) Capacity linking build and flow
    model.addConstrs(
        (f.get((i, j), 0) + f.get((j, i), 0) <=
         graph.edges[i, j]['capacity_1'] * y1[i, j] +
         graph.edges[i, j]['capacity_2'] * y2[i, j]
         for i, j in graph.edges), # Use graph.edges for unique pairs
        name="cap"
    )

    # (3) Flow conservation at each node
    for i in graph.nodes:
        # Sum of flows into node i
        # Look for variables f[j, i] where j is a neighbor of i
        inflow = gp.quicksum(f.get((j, i), 0) for j in graph.neighbors(i))

        # Sum of flows out of node i
        # Look for variables f[i, j] where j is a neighbor of i
        outflow = gp.quicksum(f.get((i, j), 0) for j in graph.neighbors(i))

        model.addConstr(
            inflow - outflow == graph.nodes[i]['supply_demand'],
            name=f"flowcons_{i}"
        )

    # Save variables for potential access later if needed (optional)
    model._y1 = y1
    model._y2 = y2
    model._f = f


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default filename relative to the project root directory
    parser.add_argument("--filename", default="instances/ex1.1-instance.dat")
    args = parser.parse_args()

    # Check if the file exists before attempting to read
    instance_path = Path(args.filename)
    if not instance_path.is_file():
        print(f"Error: Instance file not found at {instance_path}")
        print(f"Please ensure the file exists and the script is run from the project root directory.")
        exit() # Exit if file not found

    graph = read_instance_file(instance_path)

    model = gp.Model("ex1.1")
    build_model(model, graph) # Call the implemented function

    model.update()
    print("Starting Gurobi optimization...") # Added print statement
    model.optimize()
    print("Optimization finished.") # Added print statement

    # --- Output --- (Modified to use graph data and variable names)
    if model.SolCount > 0:
        print(f"\nSolver Status: {model.Status}") # Gurobi status code
        print(f"Optimal total cost: {model.ObjVal:.2f}\n")

        # Access variables using model attributes saved in build_model
        y1 = model._y1
        y2 = model._y2
        f = model._f

        print("Edges built:")
        # Iterate through the keys used for y variables (graph.edges)
        for u, v in graph.edges:
             # Use .X to get the variable value after optimization
            if y1[u, v].X > 0.5:
                print(f"  Option 1 on edge {u}-{v}")
            if y2[u, v].X > 0.5:
                print(f"  Option 2 on edge {u}-{v}")

        print("\nNonzero flows:")
        # Iterate through the flow variables dictionary f
        for (u, v), flow_var in f.items():
             # Use .X to get the variable value after optimization
            if flow_var.X > 1e-6:
                print(f"  f[{u}->{v}] = {flow_var.X:.1f}")

    elif model.Status == GRB.INFEASIBLE:
         print("Model is infeasible. Check constraints and data.")
         # Optionally compute and print IIS (Irreducible Inconsistent Subsystem)
         # model.computeIIS()
         # model.write("model_iis.ilp")
         # print("IIS written to model_iis.ilp")
    elif model.Status == GRB.UNBOUNDED:
          print("Model is unbounded. Check objective function and constraints.")
    else:
         print(f"No solution found. Status code: {model.Status}")


    model.close()