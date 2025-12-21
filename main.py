import os
import time
import csv

from config import Config
from instance import make_instance, save_instance_json, load_instance_json
from initial_solution import initial_solution
from socp_refinement import refine_by_socp, AttemptResult
from visibility import freedom_radius
from plotting import plot_iter, append_results_csv

config = Config()


def dbg(msg: str):
    if config.debug:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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

    def get_attempt_run_dir(attempt_num):
        return f"{run_dir}_attempt{attempt_num}"

    def get_attempt_route_info_dir(attempt_num):
        return f"{config.route_info_dir}_attempt{attempt_num}"

    def on_iteration(attempt_num, it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev, iter_order):
        if config.make_plots:
            attempt_dir = get_attempt_run_dir(attempt_num)
            os.makedirs(attempt_dir, exist_ok=True)
            route_info_dir = get_attempt_route_info_dir(attempt_num)
            os.makedirs(route_info_dir, exist_ok=True)
            
            fname = os.path.join(attempt_dir, f"iter_{it:02d}.png")
            plot_iter(
                inst, iter_order, Ls_now, Rs_now,
                base_verts, base_adj, Dvv, Next,
                filename=fname, show_circles=False,
                title=f"Attempt {attempt_num} Iter {it:02d}",
                obj_value=obj_value,
                freedom_range=(launch_freedom, return_freedom),
                freedom_centers=(launch_prev, return_prev),
                max_iters=config.max_iters,
                route_info_dir=route_info_dir
            )

    # Plot initial solution to attempt1 folder
    if config.make_plots:
        attempt1_dir = f"{run_dir}_attempt1"
        os.makedirs(attempt1_dir, exist_ok=True)
        attempt1_route_dir = f"{config.route_info_dir}_attempt1"
        os.makedirs(attempt1_route_dir, exist_ok=True)
        
        iter0_gtsp = os.path.join(attempt1_dir, "iter_00_gtsp.png")
        plot_iter(
            inst, order, chosen_launch, chosen_return,
            base_verts, base_adj, Dvv, Next,
            filename=iter0_gtsp, show_circles=True,
            candidate_map=res0["candidate_map"],
            title="Iter 0 — Initial GTSP solution",
            obj_value=res0["Init Obj"],
            max_iters=config.max_iters,
            route_info_dir=attempt1_route_dir
        )

        rL0 = [freedom_radius(pt, inst.obstacles, config.shrink_freedom) for pt in chosen_launch]
        rR0 = rL0[:]
        iter0_constraint = os.path.join(attempt1_dir, "iter_00_update_constraint.png")
        plot_iter(
            inst, order, chosen_launch, chosen_return,
            base_verts, base_adj, Dvv, Next,
            filename=iter0_constraint, show_circles=False,
            title="Iter 0 — Constraint regions for first update",
            obj_value=res0["Init Obj"],
            freedom_range=(rL0, rR0),
            freedom_centers=(chosen_launch, chosen_return),
            max_iters=config.max_iters,
            route_info_dir=attempt1_route_dir
        )

    dbg("  Starting SOCP iterations...")
    dbg(f"  (Will check TSP order after 25 iters, up to {config.max_reorder_attempts} restarts)")
    
    attempt_results, _ = refine_by_socp(
        inst, order, chosen_launch, chosen_return,
        max_iters=config.max_iters,
        shrink_freedom=config.shrink_freedom,
        base_verts=base_verts,
        base_adj=base_adj,
        Dvv=Dvv,
        init_obj=res0["Init Obj"],
        on_iteration=on_iteration,
        max_reorder_attempts=config.max_reorder_attempts
    )

    init_obj = res0["Init Obj"]
    
    # Find best result across all attempts
    best_attempt = min(attempt_results, key=lambda a: a.best_obj)
    best_obj = best_attempt.best_obj
    
    for ar in attempt_results:
        dbg(f"  Attempt {ar.attempt_num}: best_obj={ar.best_obj:.3f}, best_iter={ar.best_iter}, order={ar.order}")

    reduction = (init_obj - best_obj) / init_obj if init_obj > 0 else 0.0
    dbg(f"  Done: init={init_obj:.3f}, best={best_obj:.3f} (attempt {best_attempt.attempt_num}), reduction={reduction:.1%}")

    # Copy best attempt's outputs to main folder (without _attemptN suffix)
    if config.make_plots:
        import shutil
        
        best_attempt_num = best_attempt.attempt_num
        best_plots_dir = f"{run_dir}_attempt{best_attempt_num}"
        best_route_dir = f"{config.route_info_dir}_attempt{best_attempt_num}"
        
        # Copy plots
        os.makedirs(run_dir, exist_ok=True)
        if os.path.exists(best_plots_dir):
            for fname in os.listdir(best_plots_dir):
                src = os.path.join(best_plots_dir, fname)
                dst = os.path.join(run_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        # Copy route info
        os.makedirs(config.route_info_dir, exist_ok=True)
        if os.path.exists(best_route_dir):
            for fname in os.listdir(best_route_dir):
                src = os.path.join(best_route_dir, fname)
                dst = os.path.join(config.route_info_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                elif os.path.isdir(src):
                    dst_subdir = os.path.join(config.route_info_dir, fname)
                    if os.path.exists(dst_subdir):
                        shutil.rmtree(dst_subdir)
                    shutil.copytree(src, dst_subdir)

    return {
        "instance": inst,
        "init_obj": init_obj,
        "attempt_results": attempt_results,
        "best_attempt": best_attempt
    }


def append_results_csv_with_attempts(num_targets: int, num_obst: int, seed: int, 
                                      init_obj: float, attempt_results: list,
                                      results_dir: str = "."):
    """Write results CSV with columns for each attempt."""
    fname = f"results_T{num_targets}_O{num_obst}.csv"
    p = os.path.join(results_dir, fname)
    need_header = not os.path.exists(p)
    
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow([
                "seed", "initial_objective",
                "best_obj_attempt1", "order_attempt1",
                "best_obj_attempt2", "order_attempt2", 
                "best_obj_attempt3", "order_attempt3",
                "overall_best_obj", "overall_best_attempt"
            ])
        
        row = [seed, f"{init_obj:.6f}"]
        
        # Add data for each possible attempt (1, 2, 3)
        for attempt_num in [1, 2, 3]:
            matching = [a for a in attempt_results if a.attempt_num == attempt_num]
            if matching:
                ar = matching[0]
                row.append(f"{ar.best_obj:.6f}")
                row.append(str(ar.order))
            else:
                row.append("")
                row.append("")
        
        # Overall best
        best = min(attempt_results, key=lambda a: a.best_obj)
        row.append(f"{best.best_obj:.6f}")
        row.append(best.attempt_num)
        
        w.writerow(row)


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
    
    append_results_csv_with_attempts(
        num_targets, num_obst, seed,
        result["init_obj"], result["attempt_results"],
        config.results_dir
    )
    
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
