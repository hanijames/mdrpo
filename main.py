import os
import time

from config import Config
from instance import make_instance, save_instance_json, load_instance_json
from initial_solution import initial_solution
from socp_refinement import refine_by_socp
from visibility import freedom_radius
from plotting import plot_iter, append_results_csv

import csv

config = Config()


def dbg(msg: str):
    if config.debug:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_reorder_csv(num_targets: int, num_obst: int, seed: int, num_attempts: int, results_dir: str = ".", instance_name: str = None):
    fname = "reorder_attempts.csv"
    p = os.path.join(results_dir, fname)
    need_header = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["instance_name", "num_targets", "num_obstacles", "seed", "num_attempts"])
        name = instance_name if instance_name else f"T{num_targets}_O{num_obst}_S{seed}"
        w.writerow([name, num_targets, num_obst, seed, num_attempts])


def _run_instance(inst, run_dir):
    """Core solve logic shared by run_single and run_custom."""
    dbg("  Computing initial solution...")
    res0 = initial_solution(
        inst,
        ring_resolution=config.ring_resolution,
        shrink_factor=config.shrink_factor
    )
    dbg(f"  Initial solution: Init Obj={res0['Init Obj']:.3f}")

    order = res0["order"]
    chosen_launch = res0["chosen_launch"]
    chosen_return = res0["chosen_return"]
    base_verts = res0["base_verts"]
    base_adj = res0["base_adj"]
    Dvv = res0["Dvv"]
    Next = res0["Next"]

    current_order = list(order)

    if config.make_plots:
        iter0_gtsp = os.path.join(run_dir, "iter_00_gtsp.png")
        plot_iter(
            inst, order, chosen_launch, chosen_return,
            base_verts, base_adj, Dvv, Next,
            filename=iter0_gtsp, show_circles=True,
            candidate_map=res0["candidate_map"],
            title="Iter 0 — Initial GTSP solution",
            obj_value=res0["Init Obj"],
            max_iters=config.max_iters,
            route_info_dir=config.route_info_dir
        )

        rL0 = [freedom_radius(pt, inst.obstacles, config.shrink_freedom) for pt in chosen_launch]
        rR0 = rL0[:]
        iter0_constraint = os.path.join(run_dir, "iter_00_update_constraint.png")
        plot_iter(
            inst, order, chosen_launch, chosen_return,
            base_verts, base_adj, Dvv, Next,
            filename=iter0_constraint, show_circles=False,
            title="Iter 0 — Constraint regions for first update",
            obj_value=res0["Init Obj"],
            freedom_range=(rL0, rR0),
            freedom_centers=(chosen_launch, chosen_return),
            max_iters=config.max_iters,
            route_info_dir=config.route_info_dir
        )

    def on_iteration(it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev, iter_order=None):
        nonlocal current_order
        if iter_order is not None:
            current_order = list(iter_order)
        if config.make_plots:
            fname = os.path.join(run_dir, f"iter_{it:02d}.png")
            plot_iter(
                inst, current_order, Ls_now, Rs_now,
                base_verts, base_adj, Dvv, Next,
                filename=fname, show_circles=False,
                title=f"Iter {it:02d}",
                obj_value=obj_value,
                freedom_range=(launch_freedom, return_freedom),
                freedom_centers=(launch_prev, return_prev),
                max_iters=config.max_iters,
                route_info_dir=config.route_info_dir
            )

    dbg("  Starting SOCP iterations...")
    if config.max_reorder_attempts > 0:
        dbg(f"  (Will check TSP order after {config.reorder_iters} iters, up to {config.max_reorder_attempts} restarts)")
    
    progress, best_obj, best_iter, num_attempts = refine_by_socp(
        inst, order, chosen_launch, chosen_return,
        max_iters=config.max_iters,
        shrink_freedom=config.shrink_freedom,
        base_verts=base_verts,
        base_adj=base_adj,
        Dvv=Dvv,
        init_obj=res0["Init Obj"],
        on_iteration=on_iteration,
        reorder_iters=config.reorder_iters,
        max_reorder_attempts=config.max_reorder_attempts
    )

    init_obj = res0["Init Obj"]
    if not progress:
        best_obj = init_obj

    reduction = (init_obj - best_obj) / init_obj if init_obj > 0 else 0.0
    dbg(f"  Done: init={init_obj:.3f}, best={best_obj:.3f}, reduction={reduction:.1%}, attempts={num_attempts}")

    return {
        "instance": inst,
        "init_obj": init_obj,
        "best_obj": best_obj,
        "best_iter": best_iter,
        "order": current_order,
        "num_attempts": num_attempts
    }


def run_single(num_obst: int, num_targets: int, seed: int):
    dbg(f"Running T={num_targets}, O={num_obst}, seed={seed}")

    run_dir = os.path.join(config.plots_dir, f"T{num_targets}_O{num_obst}_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)

    inst_name = f"T{num_targets}_O{num_obst}_S{seed}"
    inst_path = os.path.join(config.instance_info_dir, f"{inst_name}.json")
    os.makedirs(config.instance_info_dir, exist_ok=True)

    if os.path.exists(inst_path):
        inst = load_instance_json(inst_path, alpha=config.alpha, R=config.R_endurance,
                                  orig=config.orig, dest=config.dest)
        dbg(f"  Loaded instance from {inst_path}")
    else:
        inst = make_instance(
            num_obst, num_targets, seed,
            alpha=config.alpha, R=config.R_endurance,
            orig=config.orig, dest=config.dest
        )
        save_instance_json(inst, inst_path)
        dbg(f"  Saved instance to {inst_path}")

    result = _run_instance(inst, run_dir)
    append_results_csv(num_targets, num_obst, seed, result["init_obj"], result["best_obj"], config.results_dir)
    append_reorder_csv(num_targets, num_obst, seed, result["num_attempts"], config.results_dir)
    return result


def run_custom(inst_name: str):
    dbg(f"Running custom instance: {inst_name}")

    run_dir = os.path.join(config.plots_dir, inst_name)
    os.makedirs(run_dir, exist_ok=True)

    inst_path = os.path.join(config.custom_instances_dir, f"{inst_name}.json")
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"Custom instance not found: {inst_path}")

    inst = load_instance_json(inst_path, alpha=config.alpha, R=config.R_endurance,
                              orig=config.orig, dest=config.dest)
    dbg(f"  Loaded instance from {inst_path}")

    result = _run_instance(inst, run_dir)
    append_reorder_csv(len(inst.targets), len(inst.obstacles), 0, result["num_attempts"], 
                       config.results_dir, instance_name=inst_name)
    return result


def main():
    dbg("=== RUN START ===")
    results = []

    if config.custom_instances:
        for inst_name in config.custom_instances:
            result = run_custom(inst_name)
            results.append(result)
    else:
        for num_targets in config.num_targets_list:
            for num_obst in config.num_obstacles_list:
                for seed in config.seeds:
                    result = run_single(num_obst, num_targets, seed)
                    results.append(result)

    dbg("=== RUN END ===")
    return results


if __name__ == "__main__":
    main()
