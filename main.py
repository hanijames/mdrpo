import os
import time
import csv

from config import Config
from instance import make_instance, save_instance_json, load_instance_json
from initial_solution import initial_solution
from socp_refinement import refine_by_socp, compute_total_time
from visibility import freedom_radius
from plotting import plot_iter

config = Config()


def dbg(msg: str):
    if config.debug:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def weight_to_suffix(w: float) -> str:
    """Convert weight to folder/column suffix: 1.1 -> '1p1x', 2.0 -> '2x'"""
    if w == int(w):
        return f"{int(w)}x"
    else:
        return f"{w}".replace(".", "p") + "x"


def _run_instance(inst, run_dir_base, route_info_dir_base):
    """Core solve logic with multiple ship weights."""
    dbg("  Computing initial solution...")
    res0 = initial_solution(
        inst,
        ring_resolution=config.ring_resolution,
        shrink_factor=config.shrink_factor
    )

    order = res0["order"]
    chosen_launch = res0["chosen_launch"]
    chosen_return = res0["chosen_return"]
    base_verts = res0["base_verts"]
    base_adj = res0["base_adj"]
    Dvv = res0["Dvv"]
    Next = res0["Next"]

    init_time = compute_total_time(inst, order, chosen_launch, chosen_return, base_verts, base_adj)
    dbg(f"  Initial time: {init_time:.3f}")

    all_results = {"init_time": init_time}

    for ship_weight in config.obj_weights_ship:
        suffix = weight_to_suffix(ship_weight)
        run_dir = os.path.join(run_dir_base, suffix)
        route_info_dir = os.path.join(route_info_dir_base, suffix)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(route_info_dir, exist_ok=True)

        dbg(f"  Running SOCP with ship_weight={ship_weight} ({suffix})...")

        init_obj_weighted = res0["Ship Cost"] * ship_weight + (res0["Init Obj"] - res0["Ship Cost"])

        final_points = {"Ls": list(chosen_launch), "Rs": list(chosen_return)}

        if config.make_plots:
            iter0_gtsp = os.path.join(run_dir, "iter_00_gtsp.png")
            plot_iter(
                inst, order, chosen_launch, chosen_return,
                base_verts, base_adj, Dvv, Next,
                filename=iter0_gtsp, show_circles=True,
                candidate_map=res0["candidate_map"],
                title=f"Iter 00 — GTSP",
                obj_value=init_obj_weighted,
                max_iters=config.max_iters,
                route_info_dir=route_info_dir
            )

        weight_str = int(ship_weight) if ship_weight == int(ship_weight) else ship_weight

        def on_iteration(it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev):
            final_points["Ls"] = Ls_now
            final_points["Rs"] = Rs_now
            if config.make_plots:
                fname = os.path.join(run_dir, f"iter_{it:02d}.png")
                plot_iter(
                    inst, order, Ls_now, Rs_now,
                    base_verts, base_adj, Dvv, Next,
                    filename=fname, show_circles=False,
                    title=f"Iter {it:02d} ({weight_str}x ship cost)",
                    obj_value=obj_value,
                    freedom_range=(launch_freedom, return_freedom),
                    freedom_centers=(launch_prev, return_prev),
                    max_iters=config.max_iters,
                    route_info_dir=route_info_dir
                )

        progress, best_obj, best_iter = refine_by_socp(
            inst, order, chosen_launch, chosen_return,
            max_iters=config.max_iters,
            shrink_freedom=config.shrink_freedom,
            base_verts=base_verts,
            base_adj=base_adj,
            Dvv=Dvv,
            init_obj=init_obj_weighted,
            on_iteration=on_iteration,
            stop_threshold=config.socp_stop_threshold,
            ship_weight=ship_weight
        )

        if not progress:
            best_obj = init_obj_weighted

        final_time = compute_total_time(inst, order, final_points["Ls"], final_points["Rs"], base_verts, base_adj)

        dbg(f"    {suffix}: init_obj={init_obj_weighted:.3f}, final_obj={best_obj:.3f}, final_time={final_time:.3f}")

        all_results[f"init_obj_{suffix}"] = init_obj_weighted
        all_results[f"final_obj_{suffix}"] = best_obj
        all_results[f"final_time_{suffix}"] = final_time

    return all_results


def append_results_csv_weighted(num_targets: int, num_obst: int, seed: int,
                                results: dict, results_dir: str = "."):
    """Write results CSV with columns for each weight."""
    fname = f"results_T{num_targets}_O{num_obst}.csv"
    p = os.path.join(results_dir, fname)
    need_header = not os.path.exists(p)

    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            header = ["seed", "init_time"]
            for ship_weight in config.obj_weights_ship:
                suffix = weight_to_suffix(ship_weight)
                header.extend([f"init_obj_{suffix}", f"final_obj_{suffix}", f"final_time_{suffix}"])
            w.writerow(header)

        row = [seed, f"{results['init_time']:.6f}"]
        for ship_weight in config.obj_weights_ship:
            suffix = weight_to_suffix(ship_weight)
            row.append(f"{results[f'init_obj_{suffix}']:.6f}")
            row.append(f"{results[f'final_obj_{suffix}']:.6f}")
            row.append(f"{results[f'final_time_{suffix}']:.6f}")

        w.writerow(row)


def run_single(num_obst: int, num_targets: int, seed: int):
    dbg(f"Running T={num_targets}, O={num_obst}, seed={seed}")

    run_dir_base = os.path.join(config.plots_dir, f"T{num_targets}_O{num_obst}_seed{seed}")
    route_info_dir_base = os.path.join(config.route_info_dir, f"T{num_targets}_O{num_obst}_S{seed}")

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

    results = _run_instance(inst, run_dir_base, route_info_dir_base)
    append_results_csv_weighted(num_targets, num_obst, seed, results, config.results_dir)
    return results


def run_custom(inst_name: str):
    dbg(f"Running custom instance: {inst_name}")

    run_dir_base = os.path.join(config.plots_dir, inst_name)
    route_info_dir_base = os.path.join(config.route_info_dir, inst_name)

    inst_path = os.path.join(config.custom_instances_dir, f"{inst_name}.json")
    if not os.path.exists(inst_path):
        raise FileNotFoundError(f"Custom instance not found: {inst_path}")

    inst = load_instance_json(inst_path, alpha=config.alpha, R=config.R_endurance,
                              orig=config.orig, dest=config.dest)
    dbg(f"  Loaded instance from {inst_path}")

    return _run_instance(inst, run_dir_base, route_info_dir_base)


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