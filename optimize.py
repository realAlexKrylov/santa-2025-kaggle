#!/usr/bin/env python3
"""
Santa 2025 — Christmas Tree Packing optimizer.
Adapted for RunPod / powerful CPU servers.

Usage:
    python optimize.py                         # defaults: 6h, /workspace/santa/
    python optimize.py --hours 12              # run for 12 hours
    python optimize.py --workdir ./data        # custom directory
"""

import subprocess
import shutil
import os
import sys
import time
import math
import glob
import argparse
import multiprocessing
import pandas as pd
import numpy as np
from numba import njit
from decimal import Decimal, getcontext
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

getcontext().prec = 25
scale_factor = Decimal("1e18")


# --- Numba-optimized core ---

@njit(cache=True)
def make_polygon_template():
    tw=0.15; th=0.2; bw=0.7; mw=0.4; ow=0.25
    tip=0.8; t1=0.5; t2=0.25; base=0.0; tbot=-th
    x = np.array([0,ow/2,ow/4,mw/2,mw/4,bw/2,tw/2,tw/2,-tw/2,-tw/2,-bw/2,-mw/4,-mw/2,-ow/4,-ow/2], np.float64)
    y = np.array([tip,t1,t1,t2,t2,base,base,tbot,tbot,base,base,t2,t2,t1,t1], np.float64)
    return x, y

@njit(cache=True)
def score_group_fast(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    side = max(mxx - mnx, mxy - mny)
    return side * side / n

@njit(cache=True)
def get_bounding_box(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X
            if X > mxx: mxx = X
            if Y < mny: mny = Y
            if Y > mxy: mxy = Y
    return mnx, mny, mxx, mxy

@njit(cache=True)
def find_boundary_trees(xs, ys, degs, tx, ty):
    n = xs.size
    V = tx.size
    mnx = 1e300; mny = 1e300; mxx = -1e300; mxy = -1e300
    min_x_tree = 0; min_y_tree = 0; max_x_tree = 0; max_y_tree = 0
    for i in range(n):
        r = degs[i] * math.pi / 180.0
        c = math.cos(r); s = math.sin(r)
        xi = xs[i]; yi = ys[i]
        for j in range(V):
            X = c*tx[j] - s*ty[j] + xi
            Y = s*tx[j] + c*ty[j] + yi
            if X < mnx: mnx = X; min_x_tree = i
            if X > mxx: mxx = X; max_x_tree = i
            if Y < mny: mny = Y; min_y_tree = i
            if Y > mxy: mxy = Y; max_y_tree = i
    return min_x_tree, min_y_tree, max_x_tree, max_y_tree


# --- Shapely overlap checking ---

class ChristmasTree:
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))

        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal("0.0") * scale_factor, tip_y * scale_factor),
            (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated, xoff=float(self.center_x * scale_factor), yoff=float(self.center_y * scale_factor)
        )


def create_trees_from_arrays(xs, ys, degs):
    return [ChristmasTree(str(xs[i]), str(ys[i]), str(degs[i])) for i in range(len(xs))]

def has_overlap_arrays(xs, ys, degs):
    if len(xs) <= 1:
        return False
    return has_overlap(create_trees_from_arrays(xs, ys, degs))

def has_overlap(trees):
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        for idx in tree_index.query(poly):
            if idx != i and poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False

def load_configuration_from_df(n, df):
    group_data = df[df["id"].str.startswith(f"{n:03d}_")]
    trees = []
    for _, row in group_data.iterrows():
        trees.append(ChristmasTree(row["x"][1:], row["y"][1:], row["deg"][1:]))
    return trees

def get_score(trees, n=None):
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / 1e18 for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    score = max(max_x - min_x, max_y - min_y) ** 2
    return score / n if n else score

def eval_df_sub(df, verbose=False):
    failed = []
    total_score = 0.0
    scores = {}
    for n in range(1, 201):
        trees = load_configuration_from_df(n, df)
        score = get_score(trees, n)
        scores[n] = score
        total_score += score
        if verbose:
            print(f"{n:3}  {score:.6f}")
        if has_overlap(trees):
            failed.append(n)
    if not failed:
        print("No overlaps")
    else:
        print(f"Overlaps in: {failed}")
    print(f"Score: {total_score:.12f}")
    return total_score, scores


# --- Optimization strategies ---

def simulated_annealing_config(df, config_n, max_iterations=200,
                                initial_temp=0.001, cooling_rate=0.995):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    current_score = original_score
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    temperature = initial_temp
    step_xy = 0.002 / np.sqrt(n_trees)
    step_rot = 0.5 / np.sqrt(n_trees)

    for _ in range(max_iterations):
        tree_idx = random.randint(0, n_trees - 1)
        new_xs, new_ys, new_degs = xs.copy(), ys.copy(), degs.copy()
        move = random.choice(['translate', 'rotate', 'both'])
        if move in ['translate', 'both']:
            new_xs[tree_idx] += random.gauss(0, step_xy)
            new_ys[tree_idx] += random.gauss(0, step_xy)
        if move in ['rotate', 'both']:
            new_degs[tree_idx] += random.gauss(0, step_rot)

        new_score = score_group_fast(new_xs, new_ys, new_degs, tx, ty)
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / temperature):
            if not has_overlap_arrays(new_xs, new_ys, new_degs):
                xs, ys, degs = new_xs, new_ys, new_degs
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
        temperature *= cooling_rate

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


def try_swap_trees(df, config_n, max_swaps=50):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 2:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys = xs.copy(), ys.copy()

    for _ in range(max_swaps):
        i, j = random.sample(range(n_trees), 2)
        new_xs, new_ys = xs.copy(), ys.copy()
        new_xs[i], new_xs[j] = new_xs[j], new_xs[i]
        new_ys[i], new_ys[j] = new_ys[j], new_ys[i]
        new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
        if new_score < best_score:
            if not has_overlap_arrays(new_xs, new_ys, degs):
                best_score = new_score
                best_xs, best_ys = new_xs.copy(), new_ys.copy()
                xs, ys = new_xs, new_ys

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


def optimize_boundary_trees(df, config_n, iterations=100):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    step = 0.0005
    rot_step = 0.2

    for _ in range(iterations):
        boundary_trees = set(find_boundary_trees(xs, ys, degs, tx, ty))
        for tree_idx in boundary_trees:
            mnx, mny, mxx, mxy = get_bounding_box(xs, ys, degs, tx, ty)
            cx = (mnx + mxx) / 2
            cy = (mny + mxy) / 2
            dx = cx - xs[tree_idx]
            dy = cy - ys[tree_idx]
            norm = np.sqrt(dx*dx + dy*dy)
            if norm > 1e-10:
                dx, dy = dx/norm * step, dy/norm * step

            new_xs, new_ys = xs.copy(), ys.copy()
            new_xs[tree_idx] += dx
            new_ys[tree_idx] += dy
            new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
            if new_score < best_score:
                if not has_overlap_arrays(new_xs, new_ys, degs):
                    best_score = new_score
                    xs, ys = new_xs, new_ys
                    best_xs, best_ys = xs.copy(), ys.copy()

            for drot in [-rot_step, rot_step]:
                new_degs = degs.copy()
                new_degs[tree_idx] += drot
                new_score = score_group_fast(xs, ys, new_degs, tx, ty)
                if new_score < best_score:
                    if not has_overlap_arrays(xs, ys, new_degs):
                        best_score = new_score
                        degs = new_degs
                        best_degs = degs.copy()

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


def gradient_descent_config(df, config_n, steps=30, learning_rate=0.0001):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    eps = 1e-7

    for _ in range(steps):
        grad_x = np.zeros(n_trees)
        grad_y = np.zeros(n_trees)
        for i in range(n_trees):
            xs_plus = xs.copy(); xs_plus[i] += eps
            xs_minus = xs.copy(); xs_minus[i] -= eps
            grad_x[i] = (score_group_fast(xs_plus, ys, degs, tx, ty) -
                        score_group_fast(xs_minus, ys, degs, tx, ty)) / (2 * eps)
            ys_plus = ys.copy(); ys_plus[i] += eps
            ys_minus = ys.copy(); ys_minus[i] -= eps
            grad_y[i] = (score_group_fast(xs, ys_plus, degs, tx, ty) -
                        score_group_fast(xs, ys_minus, degs, tx, ty)) / (2 * eps)

        new_xs = xs - learning_rate * grad_x
        new_ys = ys - learning_rate * grad_y
        new_score = score_group_fast(new_xs, new_ys, degs, tx, ty)
        if new_score < best_score and not has_overlap_arrays(new_xs, new_ys, degs):
            xs, ys = new_xs, new_ys
            best_score = new_score
            best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


def rotation_grid_search(df, config_n, angle_step=15):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_degs = degs.copy()
    boundary_trees = set(find_boundary_trees(xs, ys, degs, tx, ty))
    angles_to_try = np.arange(-180, 180, angle_step)

    for tree_idx in boundary_trees:
        current_best_angle = degs[tree_idx]
        for angle in angles_to_try:
            test_degs = degs.copy()
            test_degs[tree_idx] = angle
            score = score_group_fast(xs, ys, test_degs, tx, ty)
            if score < best_score:
                if not has_overlap_arrays(xs, ys, test_degs):
                    best_score = score
                    current_best_angle = angle
        degs[tree_idx] = current_best_angle
        best_degs = degs.copy()

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(xs, ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


def basin_hopping_config(df, config_n, hops=10, local_steps=50,
                         perturbation=0.04):
    tx, ty = make_polygon_template()
    group_mask = df["id"].str.startswith(f"{config_n:03d}_")
    group_data = df[group_mask].copy()
    xs = np.array([float(v[1:]) for v in group_data["x"]])
    ys = np.array([float(v[1:]) for v in group_data["y"]])
    degs = np.array([float(v[1:]) for v in group_data["deg"]])
    n_trees = len(xs)
    if n_trees <= 1:
        return False, 0

    original_score = score_group_fast(xs, ys, degs, tx, ty)
    best_score = original_score
    best_xs, best_ys, best_degs = xs.copy(), ys.copy(), degs.copy()
    perturbation_size = perturbation / np.sqrt(n_trees)

    for _ in range(hops):
        perturbed_xs = xs + np.random.uniform(-perturbation_size, perturbation_size, n_trees)
        perturbed_ys = ys + np.random.uniform(-perturbation_size, perturbation_size, n_trees)
        perturbed_degs = degs + np.random.uniform(-10, 10, n_trees)
        if has_overlap_arrays(perturbed_xs, perturbed_ys, perturbed_degs):
            continue

        local_xs, local_ys, local_degs = perturbed_xs.copy(), perturbed_ys.copy(), perturbed_degs.copy()
        local_score = score_group_fast(local_xs, local_ys, local_degs, tx, ty)
        step = 0.001 / np.sqrt(n_trees)

        for _ in range(local_steps):
            tree_idx = random.randint(0, n_trees - 1)
            for dx, dy in [(step, 0), (-step, 0), (0, step), (0, -step)]:
                test_xs, test_ys = local_xs.copy(), local_ys.copy()
                test_xs[tree_idx] += dx
                test_ys[tree_idx] += dy
                test_score = score_group_fast(test_xs, test_ys, local_degs, tx, ty)
                if test_score < local_score:
                    if not has_overlap_arrays(test_xs, test_ys, local_degs):
                        local_xs, local_ys = test_xs, test_ys
                        local_score = test_score

        if local_score < best_score:
            if not has_overlap_arrays(local_xs, local_ys, local_degs):
                best_score = local_score
                best_xs, best_ys, best_degs = local_xs.copy(), local_ys.copy(), local_degs.copy()
                xs, ys, degs = local_xs, local_ys, local_degs

    improved = best_score < original_score - 1e-15
    if improved:
        if not has_overlap_arrays(best_xs, best_ys, best_degs):
            indices = df[group_mask].index
            for i, idx in enumerate(indices):
                df.at[idx, 'x'] = f"s{best_xs[i]}"
                df.at[idx, 'y'] = f"s{best_ys[i]}"
                df.at[idx, 'deg'] = f"s{best_degs[i]}"
        else:
            improved = False
    return improved, original_score - best_score if improved else 0


# --- Adaptive parameter selector ---

class AdaptiveParameterSelector:
    def __init__(self):
        self.successes = defaultdict(int)
        self.attempts = defaultdict(int)
        self.improvement_sum = defaultdict(float)
        self.n_range = (30, 400)
        self.r_range = (10, 50)
        self.good_params = [
            (72, 34), (100, 25), (50, 30), (150, 20), (80, 35),
            (60, 40), (120, 28), (90, 32), (200, 22), (40, 38),
            (180, 18), (75, 36), (110, 26), (65, 33), (140, 24),
            (85, 30), (95, 28), (55, 35), (130, 22), (160, 20)
        ]

    def get_params(self, exploration_rate=0.25):
        if random.random() < exploration_rate or not self.successes:
            if random.random() < 0.6 and self.good_params:
                return random.choice(self.good_params)
            return (random.randint(*self.n_range), random.randint(*self.r_range))

        weights, params = [], []
        for (n, r), successes in self.successes.items():
            attempts = self.attempts[(n, r)]
            if attempts > 0:
                rate = successes / attempts
                improvement = self.improvement_sum[(n, r)] / max(attempts, 1)
                weight = (rate + 0.1) * (1 + improvement * 1e8)
                weights.append(weight)
                params.append((n, r))

        if weights:
            total = sum(weights)
            idx = random.choices(range(len(params)), [w/total for w in weights])[0]
            return params[idx]
        return self.get_params(exploration_rate=1.0)

    def record_result(self, n, r, improved, improvement=0):
        self.attempts[(n, r)] += 1
        if improved:
            self.successes[(n, r)] += 1
            self.improvement_sum[(n, r)] += improvement
            if (n, r) not in self.good_params:
                self.good_params.append((n, r))


# --- bbox3 runner ---

def run_bbox3(params, timeout=300):
    n, r = params
    try:
        subprocess.run(["./bbox3", "-n", str(n), "-r", str(r)],
                       capture_output=True, timeout=timeout)
        return (n, r, True)
    except subprocess.TimeoutExpired:
        return (n, r, False)
    except Exception:
        return (n, r, False)


# --- Main loop ---

def main():
    parser = argparse.ArgumentParser(description="Santa 2025 optimizer for RunPod")
    parser.add_argument("--workdir", default="/workspace/santa", help="Working directory with bbox3 and submission.csv")
    parser.add_argument("--hours", type=float, default=6, help="Time limit in hours")
    parser.add_argument("--bbox3-timeout", type=int, default=300, help="Timeout per bbox3 run (seconds)")
    parser.add_argument("--sa-iterations", type=int, default=300, help="Simulated annealing iterations")
    parser.add_argument("--gradient-steps", type=int, default=50, help="Gradient descent steps")
    args = parser.parse_args()

    work_dir = args.workdir
    max_time = int(args.hours * 3600)
    bbox3_timeout = args.bbox3_timeout
    sa_iterations = args.sa_iterations
    gradient_steps = args.gradient_steps

    num_cpus = multiprocessing.cpu_count()
    bbox3_parallel = max(1, num_cpus // 4)

    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    assert os.path.exists("submission.csv"), f"submission.csv not found in {work_dir}"
    assert os.path.exists("bbox3"), f"bbox3 not found in {work_dir}"
    os.chmod("bbox3", 0o755)

    print(f"CPUs: {num_cpus}, parallel bbox3: {bbox3_parallel}")
    print(f"Time limit: {args.hours}h, bbox3 timeout: {bbox3_timeout}s")

    start_time = time.time()
    df = pd.read_csv("submission.csv")
    initial_score, initial_scores = eval_df_sub(df, False)
    best_score = initial_score
    best_df = df.copy()
    best_df.to_csv("submission_backup.csv", index=False)

    param_selector = AdaptiveParameterSelector()
    sorted_configs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)
    worst_configs = [c[0] for c in sorted_configs[:60]]

    print(f"Top 5 worst configs: {worst_configs[:5]}")

    cycle = 0
    total_bbox3_improvements = 0
    total_local_improvements = 0

    while time.time() - start_time < max_time:
        cycle += 1
        elapsed = time.time() - start_time
        print(f"Cycle {cycle} ({elapsed/60:.1f}m / {max_time/60:.0f}m)")

        # Phase 1: parallel bbox3
        batch_params = [param_selector.get_params() for _ in range(bbox3_parallel)]
        with ThreadPoolExecutor(max_workers=bbox3_parallel) as pool:
            futures = list(pool.map(lambda p: run_bbox3(p, bbox3_timeout), batch_params))

        for n, r, success in futures:
            if not success:
                print(f"  bbox3 n={n} r={r}: timeout/error")

        df = pd.read_csv("submission.csv")
        new_score, _ = eval_df_sub(df, False)

        for n, r, success in futures:
            improvement = best_score - new_score
            param_selector.record_result(n, r, success and improvement > 1e-15, max(0, improvement))

        if new_score < best_score:
            improvement = best_score - new_score
            best_score = new_score
            best_df = df.copy()
            total_bbox3_improvements += 1
            print(f"  bbox3 improved: {improvement:.12f} -> {best_score:.12f}")

        # Phase 2: local optimization
        if time.time() - start_time < max_time:
            _, current_scores = eval_df_sub(df, False)
            sorted_configs = sorted(current_scores.items(), key=lambda x: x[1], reverse=True)
            worst_configs = [c[0] for c in sorted_configs[:40]]
            configs_to_optimize = random.sample(worst_configs, min(15, len(worst_configs)))

            for config_n in configs_to_optimize:
                if time.time() - start_time >= max_time:
                    break
                strategies = [
                    ('SA', lambda: simulated_annealing_config(df, config_n, sa_iterations)),
                    ('Boundary', lambda: optimize_boundary_trees(df, config_n, 120)),
                    ('Gradient', lambda: gradient_descent_config(df, config_n, gradient_steps)),
                    ('Swap', lambda: try_swap_trees(df, config_n, 50)),
                ]
                strategy_name, strategy_fn = random.choice(strategies)
                try:
                    improved, gain = strategy_fn()
                    if improved:
                        total_local_improvements += 1
                        print(f"  {strategy_name} on config {config_n}: +{gain:.12f}")
                except Exception as e:
                    print(f"  {strategy_name} error on {config_n}: {e}")

        # Phase 3: rotation (every 3 cycles)
        if cycle % 3 == 0 and time.time() - start_time < max_time:
            for config_n in random.sample(worst_configs, min(8, len(worst_configs))):
                try:
                    improved, gain = rotation_grid_search(df, config_n, angle_step=10)
                    if improved:
                        print(f"  Rotation on config {config_n}: +{gain:.12f}")
                except:
                    pass

        # Phase 4: basin hopping (every 5 cycles)
        if cycle % 5 == 0 and time.time() - start_time < max_time:
            for config_n in random.sample(worst_configs, min(5, len(worst_configs))):
                try:
                    improved, gain = basin_hopping_config(df, config_n, hops=8, local_steps=60)
                    if improved:
                        print(f"  Basin hop on config {config_n}: +{gain:.12f}")
                except:
                    pass

        df.to_csv("submission.csv", index=False)
        new_score, _ = eval_df_sub(df, False)
        if new_score < best_score:
            best_score = new_score
            best_df = df.copy()
            print(f"  New best: {best_score:.12f}")

        if cycle % 10 == 0:
            best_df.to_csv(f"submission_checkpoint_cycle{cycle}.csv", index=False)
            print(f"Checkpoint cycle {cycle}: score={best_score:.12f}, "
                  f"bbox3={total_bbox3_improvements}, local={total_local_improvements}")

    best_df.to_csv("submission.csv", index=False)
    best_df.to_csv("submission_final.csv", index=False)

    final_score, _ = eval_df_sub(best_df, False)
    print(f"Initial:  {initial_score:.12f}")
    print(f"Final:    {final_score:.12f}")
    print(f"Improved: {initial_score - final_score:.12f}")
    print(f"Cycles:   {cycle}")
    print(f"bbox3 improvements: {total_bbox3_improvements}")
    print(f"Local improvements: {total_local_improvements}")
    print(f"Total time: {(time.time()-start_time)/3600:.2f}h")


if __name__ == "__main__":
    main()
