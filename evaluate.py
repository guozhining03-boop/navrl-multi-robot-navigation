"""
Requirement #3: metrics - how we can evaluate robot performance.
Runs the pretrained NavRL policy on 6 robots / 15 obstacles in three scenario types
and reports success rate, collision rate, timeout rate and total reward.

Usage (inside WSL):
    conda activate NavRL
    cd ~/NavRL/homework
    python evaluate.py --episodes 5 --steps 350
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "..", "quick-demos"),
           os.path.join(HERE, "..", "NavRL", "quick-demos"),
           os.path.expanduser("~/NavRL/quick-demos")):
    if os.path.isdir(_p):
        sys.path.insert(0, os.path.abspath(_p))
        break

import torch  # noqa: E402
from env import sample_free_start  # noqa: E402
from utils import get_robot_state, get_ray_cast  # noqa: E402
from agent import Agent  # noqa: E402
from scenarios import build_scenario, DYN_BOUND  # noqa: E402

DT = 0.1
MAX_RANGE = 4.0
DYN_MAX_NUM = 5
HRES_DEG = 10.0
VFOV = [-10.0, 0.0, 10.0, 20.0]
GOAL_THRESHOLD = 0.3
SAFE_MARGIN = 0.3          # collision radius used by the training env
ROBOT_RADIUS = 0.3


def dyn_state_local(pos, agents_pos, agents_vel, direction, device,
                    max_range=MAX_RANGE, max_num=DYN_MAX_NUM):
    """Dynamic-obstacle observation in the goal-aligned local frame.

    Same 10-dim layout as training ([unit rel-pos xyz, dist2d, dist_z,
    local vel xyz, width, height]) but computed with an explicit 2-D rotation,
    which avoids the batch-wise division-by-zero in utils.vec_to_new_frame.
    """
    g = np.asarray(direction, dtype=float)
    g = g / max(np.linalg.norm(g), 1e-6)
    perp = np.array([-g[1], g[0]])

    rows = []
    for p, v in zip(agents_pos, agents_vel):
        rel = np.asarray(p, dtype=float) - np.asarray(pos, dtype=float)
        d = float(np.linalg.norm(rel))
        if d < 1e-4 or d > max_range:
            continue                      # skip self and out-of-range agents
        rows.append((d, rel, np.asarray(v, dtype=float)))

    rows.sort(key=lambda t: t[0])
    out = np.zeros((max_num, 10), dtype=np.float32)
    for k, (d, rel, v) in enumerate(rows[:max_num]):
        unit = np.array([rel @ g, rel @ perp]) / d
        vloc = np.array([v @ g, v @ perp])
        out[k] = [unit[0], unit[1], 0.0, d, 0.0, vloc[0], vloc[1], 0.0, 0.0, 0.0]

    return torch.tensor(out, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)


def step_world(obstacles_dyn):
    """Move dynamic obstacles at constant velocity and bounce inside the arena."""
    for obs in obstacles_dyn:
        obs["pos"] = obs["pos"] + obs["vel"] * DT
        for axis in (0, 1):
            if obs["pos"][axis] > DYN_BOUND:
                obs["pos"][axis] = DYN_BOUND
                obs["vel"][axis] *= -1.0
            elif obs["pos"][axis] < -DYN_BOUND:
                obs["pos"][axis] = -DYN_BOUND
                obs["vel"][axis] *= -1.0


def compute_reward(velocity, to_goal, range_matrix, nearest_dyn_dist, prev_velocity):
    """Same shaping scheme as isaac-training/training/scripts/env.py (_compute_step)."""
    dist = np.linalg.norm(to_goal)
    direction = to_goal / max(dist, 1e-6)
    reward_vel = float(np.dot(velocity, direction))          # progress toward goal

    d_static = np.clip(range_matrix.flatten(), 1e-6, MAX_RANGE)
    reward_safety_static = float(np.log(d_static).mean())

    d_dyn = float(np.clip(nearest_dyn_dist, 1e-6, MAX_RANGE))
    reward_safety_dynamic = float(np.log(d_dyn))

    penalty_smooth = float(np.linalg.norm(velocity - prev_velocity))

    reward = reward_vel + 1.0 + reward_safety_static + reward_safety_dynamic - 0.1 * penalty_smooth
    return reward, {
        "r_vel": reward_vel,
        "r_safety_static": reward_safety_static,
        "r_safety_dynamic": reward_safety_dynamic,
        "penalty_smooth": penalty_smooth,
    }


def run_episode(agent, scenario, seed, max_steps, device):
    rng = np.random.default_rng(seed)
    static = [(o[0], o[1], o[2]) for o in scenario["static"]]
    dynamic = [{"pos": o["pos"].copy(), "vel": o["vel"].copy(), "radius": o["radius"]}
               for o in scenario["dynamic"]]

    n = len(scenario["starts"])
    positions = [p.copy() for p in scenario["starts"]]
    velocities = [np.zeros(2) for _ in range(n)]
    goals = [g.copy() for g in scenario["goals"]]
    prev_velocities = [np.zeros(2) for _ in range(n)]

    outcome = ["timeout"] * n
    done = [False] * n
    total_reward = [0.0 for _ in range(n)]
    steps_used = [max_steps] * n
    speed_sum = [0.0 for _ in range(n)]
    speed_cnt = [0 for _ in range(n)]
    min_clearance = [np.inf for _ in range(n)]
    traj = [[p.copy()] for p in positions]
    dyn_snapshots = []
    step_rewards = np.zeros(max_steps)
    term_sums = {k: [0.0 for _ in range(n)]
                 for k in ("r_vel", "r_alive", "r_static", "r_dyn", "p_smooth")}

    for step in range(max_steps):
        obstacles_dyn = [(o["pos"][0], o["pos"][1], o["radius"]) for o in dynamic]
        all_obstacles = static + obstacles_dyn
        all_positions = positions + [o["pos"] for o in dynamic]
        all_velocities = velocities + [o["vel"] for o in dynamic]

        for i in range(n):
            if done[i]:
                continue
            pos, vel, goal = positions[i], velocities[i], goals[i]
            to_goal = goal - pos
            dist = np.linalg.norm(to_goal)
            if dist < GOAL_THRESHOLD:
                outcome[i], done[i], steps_used[i] = "success", True, step
                continue

            direction = to_goal / max(dist, 1e-6)
            state = get_robot_state(pos, goal, vel, direction, device=device)
            static_input, range_matrix, _ = get_ray_cast(
                pos, all_obstacles, max_range=MAX_RANGE, hres_deg=HRES_DEG,
                vfov_angles_deg=VFOV,
                start_angle_deg=np.degrees(np.arctan2(direction[1], direction[0])),
                device=device)
            target_tensor = torch.tensor(np.append(direction[:2], 0.0),
                                         dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
            dyn_obs_input = dyn_state_local(pos, all_positions, all_velocities,
                                            direction, device=device)
            action = agent.plan(state, static_input, dyn_obs_input, target_tensor)

            # collision checks
            hit = False
            for ox, oy, orr in all_obstacles:
                if np.hypot(pos[0] - ox, pos[1] - oy) <= orr + SAFE_MARGIN:
                    hit = True
                    break
            if not hit:
                for j, other in enumerate(positions):
                    if j == i or (done[j] and outcome[j] == "collision"):
                        continue
                    if np.hypot(pos[0] - other[0], pos[1] - other[1]) <= 2 * ROBOT_RADIUS:
                        hit = True
                        break

            nearest = np.inf
            for k, (ox, oy, orr) in enumerate(all_obstacles):
                d = np.hypot(pos[0] - ox, pos[1] - oy) - orr
                nearest = min(nearest, d)
            min_clearance[i] = min(min_clearance[i], nearest)

            reward, terms = compute_reward(action, to_goal, range_matrix,
                                           nearest if np.isfinite(nearest) else MAX_RANGE,
                                           prev_velocities[i])
            total_reward[i] += reward
            step_rewards[step] += reward
            term_sums["r_vel"][i] += terms["r_vel"]
            term_sums["r_alive"][i] += 1.0
            term_sums["r_static"][i] += terms["r_safety_static"]
            term_sums["r_dyn"][i] += terms["r_safety_dynamic"]
            term_sums["p_smooth"][i] += -0.1 * terms["penalty_smooth"]
            speed_sum[i] += float(np.linalg.norm(action))
            speed_cnt[i] += 1
            prev_velocities[i] = action.copy()

            if hit:
                outcome[i], done[i], steps_used[i] = "collision", True, step
                continue

            positions[i] = pos + action * DT
            velocities[i] = action.copy()
            traj[i].append(positions[i].copy())

        step_world(dynamic)
        if step % 20 == 0:
            dyn_snapshots.append([o["pos"].copy() for o in dynamic])
        if all(done):
            break

    return {
        "outcome": outcome,
        "total_reward": total_reward,
        "steps": steps_used,
        "avg_speed": [s / c if c else 0.0 for s, c in zip(speed_sum, speed_cnt)],
        "min_clearance": [c if np.isfinite(c) else MAX_RANGE for c in min_clearance],
        "traj": traj,
        "goals": goals,
        "static": static,
        "dyn_snapshots": dyn_snapshots,
        "step_rewards": step_rewards,
        "term_sums": term_sums,
    }


def plot_metrics(stats, out_png):
    kinds = list(stats.keys())
    labels = {"static": "Static obstacles", "dynamic": "Dynamic obstacles",
              "mixed": "Mixed (static+dynamic)"}
    metrics = [("success_rate", "Success rate"), ("collision_rate", "Collision rate"),
               ("timeout_rate", "Timeout rate")]
    x = np.arange(len(kinds))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    colors = ["#2f6f4e", "#b45309", "#4b5563"]
    for k, (key, name) in enumerate(metrics):
        vals = [stats[s][key] * 100 for s in kinds]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=name, color=colors[k])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}%",
                    ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in kinds])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("NavRL evaluation - 6 robots, 15 obstacles per scenario")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories(episodes, out_png):
    kinds = list(episodes.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150)
    for ax, kind in zip(axes, kinds):
        ep = episodes[kind]
        for ox, oy, orr in ep["static"]:
            ax.add_patch(Circle((ox, oy), orr, color="#9ca3af"))
        for snap in ep["dyn_snapshots"][:2]:
            for p in snap:
                ax.add_patch(Circle((p[0], p[1]), 0.5, color="#f59e0b", alpha=0.35,
                                    linestyle="--", fill=False))
        cmap = cm.get_cmap("tab10", len(ep["traj"]))
        for i, (tr, goal) in enumerate(zip(ep["traj"], ep["goals"])):
            tr = np.array(tr)
            ax.plot(tr[:, 0], tr[:, 1], color=cmap(i), lw=1.4,
                    label=f"R{i+1}")
            ax.plot(goal[0], goal[1], "*", color=cmap(i), ms=9)
            ax.plot(tr[0, 0], tr[0, 1], "s", color=cmap(i), ms=5)
        ax.set_xlim(-19, 19)
        ax.set_ylim(-19, 19)
        ax.set_aspect("equal")
        ax.set_title({"static": "a) Static obstacles",
                      "dynamic": "b) Dynamic obstacles",
                      "mixed": "c) Mixed obstacles"}[kind], fontsize=11)
        ax.set_xlabel("x position (m)")
        ax.set_ylabel("y position (m)")
        ax.grid(linestyle=":", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_demo_gif(episodes, out_gif, fps=10, frame_step=3):
    """Animate the first episode of each scenario along the recorded trajectories."""
    kinds = list(episodes.keys())
    titles = {"static": "a) Static obstacles", "dynamic": "b) Dynamic obstacles",
              "mixed": "c) Mixed obstacles"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=100)

    data = {}
    max_len = 1
    for kind, ep in episodes.items():
        traj = [np.array(tr) for tr in ep["traj"]]
        max_len = max(max_len, max(len(t) for t in traj))
        cmap = cm.get_cmap("tab10", len(traj))
        data[kind] = {"traj": traj, "goals": ep["goals"], "static": ep["static"],
                      "snaps": ep["dyn_snapshots"], "cmap": cmap}

    def update(frame):
        for ax, kind in zip(axes, kinds):
            ax.clear()
            d = data[kind]
            for ox, oy, orr in d["static"]:
                ax.add_patch(Circle((ox, oy), orr, color="#9ca3af"))
            if d["snaps"]:
                snap = d["snaps"][min(frame // 20, len(d["snaps"]) - 1)]
                for p in snap:
                    ax.add_patch(Circle((p[0], p[1]), 0.5, color="#f59e0b",
                                        alpha=0.35, linestyle="--", fill=False))
            for i, tr in enumerate(d["traj"]):
                upto = min(frame + 1, len(tr))
                ax.plot(tr[:upto, 0], tr[:upto, 1], color=d["cmap"](i), lw=1.3)
                ax.plot(tr[upto - 1, 0], tr[upto - 1, 1], "o",
                        color=d["cmap"](i), ms=5)
                ax.plot(d["goals"][i][0], d["goals"][i][1], "*",
                        color=d["cmap"](i), ms=9)
            ax.set_xlim(-19, 19)
            ax.set_ylim(-19, 19)
            ax.set_aspect("equal")
            ax.set_title(titles[kind], fontsize=11)
            ax.set_xlabel("x position (m)")
            ax.set_ylabel("y position (m)")
            ax.grid(linestyle=":", alpha=0.35)
            ax.spines[["top", "right"]].set_visible(False)

    frames = list(range(0, max_len, frame_step))
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_reward_curve(episodes, out_png):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    colors = {"static": "#2f6f4e", "dynamic": "#b45309", "mixed": "#4b5563"}
    for kind, ep in episodes.items():
        r = ep["step_rewards"]
        nz = np.nonzero(r)[0]
        upto = nz[-1] + 1 if len(nz) else 1
        ax.plot(np.arange(upto), r[:upto], color=colors[kind], lw=1.3,
                label={"static": "Static", "dynamic": "Dynamic",
                       "mixed": "Mixed"}[kind])
    ax.axhline(0, color="#111827", lw=0.8, ls="--")
    ax.set_xlabel("simulation step (dt = 0.1 s)")
    ax.set_ylabel("summed reward per step (6 robots)")
    ax.set_title("Instantaneous reward during one episode")
    ax.legend(frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    device = torch.device("cpu")
    agent = Agent(device=device)

    stats = {}
    episodes = {}
    rows = []
    t0 = time.time()

    for kind in ("static", "dynamic", "mixed"):
        succ = coll = timeo = 0
        n_agents = 0
        rewards = []
        speeds = []
        clears = []
        steps_list = []
        term_lists = {k: [] for k in ("r_vel", "r_alive", "r_static", "r_dyn", "p_smooth")}
        for e in range(args.episodes):
            sc = build_scenario(kind, seed=100 + e)
            ep = run_episode(agent, sc, seed=100 + e, max_steps=args.steps, device=device)
            for i, out in enumerate(ep["outcome"]):
                n_agents += 1
                succ += out == "success"
                coll += out == "collision"
                timeo += out == "timeout"
                rows.append({
                    "scenario": kind, "episode": e, "robot": i + 1, "outcome": out,
                    "total_reward": round(ep["total_reward"][i], 3),
                    "steps": ep["steps"][i],
                    "avg_speed": round(ep["avg_speed"][i], 3),
                    "min_clearance": round(ep["min_clearance"][i], 3),
                })
                for k in term_lists:
                    term_lists[k].append(ep["term_sums"][k][i])
            rewards.append(sum(ep["total_reward"]))
            speeds.extend(ep["avg_speed"])
            clears.extend(ep["min_clearance"])
            steps_list.extend(ep["steps"])
            if e == 0:
                episodes[kind] = ep
            print(f"[{kind}] episode {e+1}/{args.episodes} outcomes={ep['outcome']} "
                  f"reward={sum(ep['total_reward']):.2f}")
        breakdown = {k: float(np.mean(v)) for k, v in term_lists.items()}
        stats[kind] = {
            "success_rate": succ / n_agents,
            "collision_rate": coll / n_agents,
            "timeout_rate": timeo / n_agents,
            "mean_episode_reward": float(np.mean(rewards)),
            "mean_robot_reward": float(np.mean([r for r in rewards]) / 6),
            "mean_avg_speed": float(np.mean(speeds)),
            "mean_min_clearance": float(np.mean(clears)),
            "mean_steps": float(np.mean(steps_list)),
            "robots_evaluated": n_agents,
            "reward_breakdown": breakdown,
        }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    plot_metrics(stats, os.path.join(args.out, "metrics_bar.png"))
    plot_trajectories(episodes, os.path.join(args.out, "trajectories.png"))
    plot_reward_curve(episodes, os.path.join(args.out, "reward_curve.png"))
    plot_demo_gif(episodes, os.path.join(args.out, "demo.gif"))

    print("\n=== summary (%.1f s) ===" % (time.time() - t0))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
