import os
import time
import csv
import shutil

from config import Config
from instance import make_instance, save_instance_json, load_instance_json
from initial_solution import initial_solutions_multi
from socp_refinement import refine_by_socp
from visibility import freedom_radius
from plotting import plot_iter

config = Config()


def dbg(msg: str):
    if config.debug:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _run_instance(inst, run_dir):
    """Core solve logic with multi-start GTSP."""
    dbg("  Computing initial solutions with multiple seeds...")
    solutions = initial_solutions_multi(
        inst,
        ring_resolution=config.ring_resolution,
        shrink_factor=config.shrink_factor,
        max_seeds=config.num_gtsp_seeds,
        target_starts=config.num_starts
    )
    dbg(f"  Found {len(solutions)} distinct GTSP solutions")
    
    for i, sol in enumerate(solutions):
        dbg(f"    Start {i+1}: seed={sol.get('seed', '?')}, Init Obj={sol['Init Obj']:.3f}")
    
    # Shared across all solutions
    base_verts = solutions[0]["base_verts"]
    base_adj = solutions[0]["base_adj"]
    Dvv = solutions[0]["Dvv"]
    Next = solutions[0]["Next"]
    candidate_map = solutions[0]["candidate_map"]
    
    all_results = []
    
    for sol_idx, sol in enumerate(solutions):
        start_num = sol_idx + 1
        start_dir = os.path.join(run_dir, f"start{start_num}")
        start_route_dir = os.path.join(config.route_info_dir, f"start{start_num}")
        os.makedirs(start_dir, exist_ok=True)
        os.makedirs(start_route_dir, exist_ok=True)
        
        order = sol["order"]
        chosen_launch = sol["chosen_launch"]
        chosen_return = sol["chosen_return"]
        init_obj = sol["Init Obj"]
        
        dbg(f"  Running SOCP for start {start_num}...")
        
        # Track the last iteration's filename for copying to _final
        last_iter_file = [None]
        
        if config.make_plots:
            iter0_gtsp = os.path.join(start_dir, "iter_00_gtsp.png")
            plot_iter(
                inst, order, chosen_launch, chosen_return,
                base_verts, base_adj, Dvv, Next,
                filename=iter0_gtsp, show_circles=True,
                candidate_map=candidate_map,
                title=f"Start {start_num} Iter 00 — GTSP (seed {sol.get('seed', '?')})",
                obj_value=init_obj,
                max_iters=config.max_iters,
                route_info_dir=start_route_dir
            )
            
            rL0 = [freedom_radius(pt, inst.obstacles, config.shrink_freedom) for pt in chosen_launch]
            rR0 = rL0[:]
            iter0_constraint = os.path.join(start_dir, "iter_00_update_constraint.png")
            plot_iter(
                inst, order, chosen_launch, chosen_return,
                base_verts, base_adj, Dvv, Next,
                filename=iter0_constraint, show_circles=False,
                title=f"Start {start_num} Iter 00 — Constraint regions",
                obj_value=init_obj,
                freedom_range=(rL0, rR0),
                freedom_centers=(chosen_launch, chosen_return),
                max_iters=config.max_iters,
                route_info_dir=start_route_dir
            )
        
        def on_iteration(it, obj_value, Ls_now, Rs_now, launch_freedom, return_freedom, launch_prev, return_prev):
            if config.make_plots:
                fname = os.path.join(start_dir, f"iter_{it:02d}.png")
                plot_iter(
                    inst, order, Ls_now, Rs_now,
                    base_verts, base_adj, Dvv, Next,
                    filename=fname, show_circles=False,
                    title=f"Start {start_num} Iter {it:02d}",
                    obj_value=obj_value,
                    freedom_range=(launch_freedom, return_freedom),
                    freedom_centers=(launch_prev, return_prev),
                    max_iters=config.max_iters,
                    route_info_dir=start_route_dir
                )
                last_iter_file[0] = fname
        
        progress, best_obj, best_iter = refine_by_socp(
            inst, order, chosen_launch, chosen_return,
            max_iters=config.max_iters,
            shrink_freedom=config.shrink_freedom,
            base_verts=base_verts,
            base_adj=base_adj,
            Dvv=Dvv,
            init_obj=init_obj,
            on_iteration=on_iteration,
            stop_threshold=config.socp_stop_threshold
        )
        
        if not progress:
            best_obj = init_obj
        
        # Copy last iteration figure to _final.png
        if config.make_plots and last_iter_file[0] is not None:
            final_file = os.path.join(start_dir, "iter_final.png")
            shutil.copy2(last_iter_file[0], final_file)
        
        num_iters = len(progress)
        reduction = (init_obj - best_obj) / init_obj if init_obj > 0 else 0.0
        dbg(f"    Start {start_num}: init={init_obj:.3f}, best={best_obj:.3f}, iters={num_iters}, reduction={reduction:.1%}")
        
        all_results.append({
            "start_num": start_num,
            "seed": sol.get("seed", 0),
            "init_obj": init_obj,
            "best_obj": best_obj,
            "best_iter": best_iter,
            "num_iters": num_iters,
            "order": order,
            "start_dir": start_dir,
            "start_route_dir": start_route_dir
        })
    
    # Find overall best
    best_result = min(all_results, key=lambda r: r["best_obj"])
    dbg(f"  Overall best: Start {best_result['start_num']} with obj={best_result['best_obj']:.3f}")
    
    # Copy best result to main folder (without _startX suffix)
    if config.make_plots:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(config.route_info_dir, exist_ok=True)
        
        best_start_dir = best_result["start_dir"]
        best_route_dir = best_result["start_route_dir"]
        
        # Copy plots
        if os.path.exists(best_start_dir):
            for fname in os.listdir(best_start_dir):
                src = os.path.join(best_start_dir, fname)
                dst = os.path.join(run_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
        
        # Copy route info
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
        "all_results": all_results,
        "best_result": best_result,
        "num_starts": len(solutions)
    }


def append_results_csv_multi(num_targets: int, num_obst: int, seed: int,
                              all_results: list, results_dir: str = "."):
    """Write results CSV with columns for each starting point."""
    fname = f"results_T{num_targets}_O{num_obst}.csv"
    p = os.path.join(results_dir, fname)
    need_header = not os.path.exists(p)
    
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            header = ["seed"]
            for i in range(1, len(all_results)+1):  # Up to 10 starts
                header.extend([f"init_obj_start{i}", f"best_obj_start{i}", f"order_start{i}"])
            header.extend(["overall_best_obj", "overall_best_start"])
            w.writerow(header)
        
        row = [seed]
        
        # Add data for each possible start (1-5)
        for start_num in range(1, len(all_results)+1):
            matching = [r for r in all_results if r["start_num"] == start_num]
            if matching:
                r = matching[0]
                row.append(f"{r['init_obj']:.6f}")
                row.append(f"{r['best_obj']:.6f}")
                row.append(str(r['order']))
            else:
                row.extend(["", "", ""])
        
        # Overall best
        best = min(all_results, key=lambda r: r["best_obj"])
        row.append(f"{best['best_obj']:.6f}")
        row.append(best["start_num"])
        
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
    
    append_results_csv_multi(
        num_targets, num_obst, seed,
        result["all_results"],
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

    return _run_instance(inst, run_dir)


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
