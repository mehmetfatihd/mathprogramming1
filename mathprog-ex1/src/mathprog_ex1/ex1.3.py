import argparse

import gurobipy as gp
from gurobipy import GRB


def build_model(model: gp.Model, n: int, k: int):
    n = int(n)
    k = int(k)
    M = 1000  # Big-M for logical implication

    # Variables
    w = model.addVars(n, n, vtype=GRB.BINARY, name="win")    # i beats j
    d = model.addVars(n, n, vtype=GRB.BINARY, name="draw")   # i draws with j
    p = model.addVars(n, vtype=GRB.INTEGER, name="points")   # total points
    r = model.addVars(n, vtype=GRB.BINARY, name="relegated") # is relegated
    P = model.addVar(vtype=GRB.INTEGER, name="min_safe_points")  # min points to be safe

    # Store variables on the model object for external access
    model._w = w
    model._d = d
    model._p = p
    model._r = r
    model._P = P

    # Each match (i vs j) has one outcome
    for i in range(n):
        for j in range(n):
            if i != j:
                model.addConstr(w[i, j] + d[i, j] <= 1, name=f"match_outcome_{i}_{j}")

    # Points calculation
    for i in range(n):
        model.addConstr(
            p[i] ==
            3 * gp.quicksum(w[i, j] for j in range(n) if j != i) +
            gp.quicksum(d[i, j] + d[j, i] for j in range(n) if j != i),
            name=f"points_calc_{i}"
        )

    # Total points constraint
    model.addConstr(gp.quicksum(p[i] for i in range(n)) == 3 * n * (n - 1), name="total_points")

    # Relegation count
    model.addConstr(gp.quicksum(r[i] for i in range(n)) == k, name="relegation_count")

    # Safe teams must have at least P points
    for i in range(n):
        model.addConstr(p[i] >= P - M * r[i], name=f"min_points_safe_{i}")

    # Objective: maximize minimum guaranteed points to avoid relegation
    model.setObjective(P, GRB.MAXIMIZE)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", default=18)
    parser.add_argument("--k", default=3)
    args = parser.parse_args()

    model = gp.Model("ex1.3")
    build_model(model, args.n, args.k)

    model.update()
    model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
