import argparse
import os
from pathlib import Path

import gurobipy as gp
import numpy as np
from gurobipy import GRB


def read_instance_file(filename: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    with open(Path(filename), mode="r", encoding="utf-8") as f:
        n_jobs = int(f.readline())
        n_machines = int(f.readline())

        # skip comment line
        f.readline()

        proc_times = []
        for _ in range(n_jobs):
            proc_times_j = [int(p) for p in f.readline().split()]
            assert len(proc_times_j) == n_machines
            proc_times.append(proc_times_j)
        processing_times = np.array(proc_times, dtype=np.int32)

        # skip comment line
        f.readline()
        machine_seq = []
        for _ in range(n_jobs):
            machine_seq_j = [int(h) for h in f.readline().split()]
            assert set(machine_seq_j) == set(range(n_machines))
            machine_seq.append(machine_seq_j)
        machine_sequences = np.array(machine_seq, dtype=np.int32)

        return processing_times, machine_sequences


def build_model(model: gp.Model, processing_times: np.ndarray, machine_sequences: np.ndarray):
    # note that both jobs and machines are 0-indexed here
    n_jobs, n_machines = processing_times.shape

    big_M = np.sum(processing_times)

    # --- Decision Variables ---
    S = model.addVars(n_jobs, n_machines, vtype=GRB.CONTINUOUS, name="S", lb=0.0)

    C = model.addVars(n_jobs, vtype=GRB.CONTINUOUS, name="C", lb=0.0)
    x = model.addVars(
        (
            (i, j, h)
            for i in range(n_jobs)
            for j in range(i + 1, n_jobs)
            for h in range(n_machines)
        ),
        vtype=GRB.BINARY,
        name="x",
    )

    model._S = S
    model._C = C
    model._x = x


    model.setObjective(gp.quicksum(C[j] for j in range(n_jobs)), GRB.MINIMIZE)

    # --- Constraints ---

    for j in range(n_jobs):
        for k in range(n_machines - 1):
            current_machine = machine_sequences[j, k]
            next_machine = machine_sequences[j, k + 1]
            proc_time_current = processing_times[j, current_machine]

            model.addConstr(
                S[j, next_machine] >= S[j, current_machine] + proc_time_current,
                name=f"prec_{j}_{k}" 
            )

    for h in range(n_machines):
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs): 
            
                proc_time_i = processing_times[i, h]

                proc_time_j = processing_times[j, h]


                model.addConstr(
                    S[j, h] >= S[i, h] + proc_time_i - big_M * (1 - x[i, j, h]),
                    name=f"disj1_{i}_{j}_{h}" 
                )

                model.addConstr(
                    S[i, h] >= S[j, h] + proc_time_j - big_M * x[i, j, h],
                     name=f"disj2_{i}_{j}_{h}"
                )


    for j in range(n_jobs):
        last_machine = machine_sequences[j, n_machines - 1]
        proc_time_last = processing_times[j, last_machine]

        model.addConstr(
            C[j] >= S[j, last_machine] + proc_time_last,
             name=f"completion_{j}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default="instances/ex1.2-instance.dat")
    args = parser.parse_args()

    processing_times, machine_sequences = read_instance_file(args.filename)
    n_jobs, n_machines = processing_times.shape

    model = gp.Model("ex1.2")
    build_model(model, processing_times, machine_sequences)

    model.update()
    model.optimize()

    if model.SolCount > 0:
        print(f"obj. value = {model.ObjVal}")
        for v in model.getVars():
            print(f"{v.VarName} = {v.X}")

    model.close()
