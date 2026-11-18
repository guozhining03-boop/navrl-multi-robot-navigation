"""Self-check over results/metrics.json.

Verifies:
  * success + collision + timeout rates sum to 1 per scenario;
  * 60 robot runs were evaluated per scenario;
  * the five active reward terms sum to the mean reward per robot.

Run from the repository root:
    python tests/check_metrics.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRICS = os.path.join(ROOT, "results", "metrics.json")

TOL_RATE = 1e-9
TOL_REWARD = 1.0  # reward sums are compared per robot, 1.0 tolerance


def main():
    with open(METRICS, "r") as f:
        stats = json.load(f)

    for scenario, data in stats.items():
        rates = data["success_rate"] + data["collision_rate"] + data["timeout_rate"]
        assert abs(rates - 1.0) < TOL_RATE, (
            f"{scenario}: rates sum to {rates}, expected 1.0"
        )
        assert data["robots_evaluated"] == 60, (
            f"{scenario}: robots_evaluated = {data['robots_evaluated']}, expected 60"
        )

        bd = data["reward_breakdown"]
        breakdown_sum = (
            bd["r_vel"] + bd["r_alive"] + bd["r_static"] + bd["r_dyn"] + bd["p_smooth"]
        )
        target = data["mean_robot_reward"]
        assert abs(breakdown_sum - target) < TOL_REWARD, (
            f"{scenario}: breakdown sum {breakdown_sum:.2f} != mean robot reward "
            f"{target:.2f}"
        )
        print(f"[OK] {scenario}: rates sum to 1.0, 60 runs, "
              f"breakdown {breakdown_sum:.2f} ~= robot reward {target:.2f}")

    print("All metric checks passed.")


if __name__ == "__main__":
    main()
