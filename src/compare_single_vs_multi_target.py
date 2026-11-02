import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Union

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multi_target_environment import MultiTargetArenaEnv, STATE_DIM, PRESET_TARGETS
from evaluate_multi_target import evaluate_matchup, get_scripted_defender_action
from agents.ppo import PPOAgent


def to_serializable(obj):
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(to_serializable(k)): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj


def load_ppo_agent(model_path: str, state_dim: int, action_dim: int = 5) -> Optional[PPOAgent]:
    """Helper to load a PPOAgent checkpoint safely."""
    if not os.path.exists(model_path):
        return None
    agent = PPOAgent(state_dim, action_dim)
    try:
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        agent.policy.load_state_dict(state_dict)
        agent.policy_old.load_state_dict(state_dict)
        agent.policy.eval()
        agent.policy_old.eval()
        return agent
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None


def run_comprehensive_comparison(
    k_values: list = [1, 2, 3, 5],
    num_episodes: int = 500,
    models_dir: str = "models/multi_target",
    phase4_model_path: str = "models/phase4_attacker.pth",
    results_dir: str = "results/multi_target",
    seed_base: int = 42,
):
    """
    Executes the paired scientific evaluation suite across single- and multi-target setups.
    """
    os.makedirs(results_dir, exist_ok=True)
    seeds = [seed_base + i for i in range(num_episodes)]

    archetypes = [
        "learned_defender",
        "stay",
        "random",
        "pursuit",
        "north_camper",
        "center_camper",
        "south_camper",
        "patroller",
    ]

    all_results = {}
    csv_rows = []

    print("=" * 80)
    print(f"  SCIENTIFIC COMPARISON: SINGLE-TARGET VS MULTI-TARGET ARENA")
    print(f"  Target Counts: {k_values} | Paired Test Episodes: {num_episodes}")
    print(f"  Random Seed Range: [{seeds[0]} ... {seeds[-1]}]")
    print("=" * 80)

    # Check for baseline Phase 4 6D attacker
    phase4_att = load_ppo_agent(phase4_model_path, state_dim=6) if os.path.exists(phase4_model_path) else None
    if phase4_att:
        print(f"[Baseline] Found Phase 4 single-target model: {phase4_model_path}")
    else:
        print(f"[Notice] Single-target baseline {phase4_model_path} not found. Transfer comparison will be skipped.")

    for k in k_values:
        env = MultiTargetArenaEnv(num_targets=k)
        all_results[f"K_{k}"] = {}

        # Load trained multi-target agents for this K
        att_path = os.path.join(models_dir, f"ppo_k{k}_attacker.pth")
        def_path = os.path.join(models_dir, f"ppo_k{k}_defender.pth")

        marl_attacker = load_ppo_agent(att_path, state_dim=STATE_DIM)
        marl_defender = load_ppo_agent(def_path, state_dim=STATE_DIM)

        print(f"\n--- Benchmarking K={k} Target(s) ---")
        if not marl_attacker:
            print(f"  [Warning] Attacker model {att_path} not found! Evaluation will use random attacker.")

        for arch in archetypes:
            def_opponent = marl_defender if arch == "learned_defender" else arch
            if arch == "learned_defender" and not marl_defender:
                print("  [Skip] Learned defender model missing for K=", k)
                continue

            att_policy = marl_attacker if marl_attacker is not None else "random"
            metrics = evaluate_matchup(
                env=env,
                attacker=att_policy,
                defender=def_opponent,
                num_episodes=num_episodes,
                seeds=seeds,
                is_attacker_6d=False,
            )

            all_results[f"K_{k}"][arch] = metrics
            print(
                f"  Def: {arch:<18} | Att Win: {metrics['attacker_win_rate']*100:5.1f}% (±{metrics['attacker_ci_95']*100:4.1f}%) "
                f"| Def Win: {metrics['defender_win_rate']*100:5.1f}% | Steps: {metrics['avg_episode_steps']:5.1f}"
            )

            csv_rows.append({
                "k_targets": k,
                "attacker_type": "marl_ppo_19d" if marl_attacker else "random",
                "defender_type": arch,
                "episodes": num_episodes,
                "att_win_rate": round(metrics["attacker_win_rate"], 4),
                "att_ci_95": round(metrics["attacker_ci_95"], 4),
                "def_win_rate": round(metrics["defender_win_rate"], 4),
                "def_ci_95": round(metrics["defender_ci_95"], 4),
                "timeout_rate": round(metrics["timeout_rate"], 4),
                "avg_steps": round(metrics["avg_episode_steps"], 2),
                "sem_steps": round(metrics["sem_episode_steps"], 3),
            })

        # Also evaluate zero-shot transfer of baseline single-target 6D model if present
        if phase4_att:
            transfer_metrics = evaluate_matchup(
                env=env,
                attacker=phase4_att,
                defender="pursuit",
                num_episodes=num_episodes,
                seeds=seeds,
                is_attacker_6d=True,
            )
            all_results[f"K_{k}"]["transfer_phase4_vs_pursuit"] = transfer_metrics
            csv_rows.append({
                "k_targets": k,
                "attacker_type": "baseline_phase4_6d_transfer",
                "defender_type": "pursuit",
                "episodes": num_episodes,
                "att_win_rate": round(transfer_metrics["attacker_win_rate"], 4),
                "att_ci_95": round(transfer_metrics["attacker_ci_95"], 4),
                "def_win_rate": round(transfer_metrics["defender_win_rate"], 4),
                "def_ci_95": round(transfer_metrics["defender_ci_95"], 4),
                "timeout_rate": round(transfer_metrics["timeout_rate"], 4),
                "avg_steps": round(transfer_metrics["avg_episode_steps"], 2),
                "sem_steps": round(transfer_metrics["sem_episode_steps"], 3),
            })

    # Save JSON and CSV
    json_path = os.path.join(results_dir, "comparison_metrics.json")
    with open(json_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\nSaved raw scientific metrics to {json_path}")

    csv_path = os.path.join(results_dir, "comparison_metrics.csv")
    if csv_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved metric tabular summary to {csv_path}")

    # Generate Publication-Quality Figures
    generate_comparison_figures(all_results, k_values, results_dir)


def generate_comparison_figures(all_results: dict, k_values: list, results_dir: str):
    """Generates all comparative charts required for scientific analysis."""
    # Palette
    colors = {
        "learned_defender": "#1d3557",
        "pursuit": "#e63946",
        "patroller": "#2a9d8f",
        "center_camper": "#e76f51",
        "north_camper": "#457b9d",
        "south_camper": "#8338ec",
        "stay": "#6c757d",
        "random": "#a8dadc",
    }

    # 1. Attacker Win Rate vs Target Count K
    plt.figure(figsize=(9.0, 5.0))
    eval_archetypes = ["learned_defender", "pursuit", "patroller", "center_camper", "stay"]
    for arch in eval_archetypes:
        x_vals = []
        y_vals = []
        y_errs = []
        for k in k_values:
            k_key = f"K_{k}"
            if k_key in all_results and arch in all_results[k_key]:
                m = all_results[k_key][arch]
                x_vals.append(k)
                y_vals.append(m["attacker_win_rate"] * 100.0)
                y_errs.append(m["attacker_ci_95"] * 100.0)
        if x_vals:
            lbl = arch.replace("_", " ").title()
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs, marker="o", linewidth=2.0, capsize=4,
                label=f"vs {lbl}", color=colors.get(arch, "#333333")
            )

    plt.title("Attacker Win Rate vs Target Count (K ∈ {1, 2, 3, 5})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Targets (K)", fontsize=11)
    plt.ylabel("Attacker Win Rate (%) [with 95% CI]", fontsize=11)
    plt.xticks(k_values)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9.5)
    plt.tight_layout()
    plot1 = os.path.join(results_dir, "attacker_win_rate_vs_targets.png")
    plt.savefig(plot1, dpi=120)
    plt.close()
    print(f"Generated chart: {plot1}")

    # 2. Defender Capture Rate vs Target Count K
    plt.figure(figsize=(9.0, 5.0))
    for arch in ["learned_defender", "pursuit", "patroller", "center_camper"]:
        x_vals = []
        y_vals = []
        y_errs = []
        for k in k_values:
            k_key = f"K_{k}"
            if k_key in all_results and arch in all_results[k_key]:
                m = all_results[k_key][arch]
                x_vals.append(k)
                y_vals.append(m["defender_win_rate"] * 100.0)
                y_errs.append(m["defender_ci_95"] * 100.0)
        if x_vals:
            lbl = arch.replace("_", " ").title()
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs, marker="s", linewidth=2.0, capsize=4,
                label=f"{lbl}", color=colors.get(arch, "#333333")
            )

    plt.title("Defender Capture Rate vs Target Count (K ∈ {1, 2, 3, 5})", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Targets (K)", fontsize=11)
    plt.ylabel("Defender Capture Rate (%) [with 95% CI]", fontsize=11)
    plt.xticks(k_values)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9.5)
    plt.tight_layout()
    plot2 = os.path.join(results_dir, "defender_capture_rate_vs_targets.png")
    plt.savefig(plot2, dpi=120)
    plt.close()
    print(f"Generated chart: {plot2}")

    # 3. Episode Duration vs Target Count K
    plt.figure(figsize=(9.0, 4.8))
    for arch in ["learned_defender", "pursuit", "patroller"]:
        x_vals = []
        y_vals = []
        y_errs = []
        for k in k_values:
            k_key = f"K_{k}"
            if k_key in all_results and arch in all_results[k_key]:
                m = all_results[k_key][arch]
                x_vals.append(k)
                y_vals.append(m["avg_episode_steps"])
                y_errs.append(m["sem_episode_steps"])
        if x_vals:
            lbl = arch.replace("_", " ").title()
            plt.errorbar(
                x_vals, y_vals, yerr=y_errs, marker="^", linewidth=2.0, capsize=4,
                label=f"vs {lbl}", color=colors.get(arch, "#333333")
            )

    plt.title("Episode Duration vs Target Count (Mean Steps ± 1 SEM)", fontsize=12, fontweight="bold")
    plt.xlabel("Number of Targets (K)", fontsize=11)
    plt.ylabel("Average Episode Steps", fontsize=11)
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=9.5)
    plt.tight_layout()
    plot3 = os.path.join(results_dir, "episode_duration_vs_targets.png")
    plt.savefig(plot3, dpi=120)
    plt.close()
    print(f"Generated chart: {plot3}")

    # 4. Target Choice Distribution for Multi-Target Configurations (K=2, 3, 5)
    multi_k = [k for k in k_values if k > 1]
    if multi_k:
        fig, axes = plt.subplots(1, len(multi_k), figsize=(4.0 * len(multi_k), 4.2), sharey=True)
        if len(multi_k) == 1:
            axes = [axes]

        for i, k in enumerate(multi_k):
            ax = axes[i]
            k_key = f"K_{k}"
            # Use learned defender or pursuit
            target_metric = None
            if k_key in all_results:
                if "learned_defender" in all_results[k_key]:
                    target_metric = all_results[k_key]["learned_defender"]
                elif "pursuit" in all_results[k_key]:
                    target_metric = all_results[k_key]["pursuit"]

            if target_metric and "target_breach_distribution" in target_metric:
                dist = target_metric["target_breach_distribution"]
                target_coords = target_metric.get("target_coords", [])
                
                # Format clean, human-readable labels such as "Target 0\n(10, 1)" without np.int64
                labels = []
                for idx in dist.keys():
                    t_idx = int(idx)
                    coord_str = ""
                    if t_idx < len(target_coords):
                        coord = target_coords[t_idx]
                        coord_str = f"\n({int(coord[0])}, {int(coord[1])})"
                    labels.append(f"Target {t_idx}{coord_str}")

                counts = list(dist.values())
                total_hits = sum(counts)
                fractions = [(c / total_hits * 100.0) if total_hits > 0 else 0.0 for c in counts]
                bars = ax.bar(labels, fractions, color="#2b5c8f", edgecolor="#1d3557", alpha=0.85)
                n_eps = target_metric.get("num_episodes", 500)
                ax.set_title(f"K = {k} Targets ({total_hits} Breaches / {n_eps} Eps)", fontsize=10.5, fontweight="bold")
                ax.set_ylabel("Share of Successful Breaches (%)" if i == 0 else "", fontsize=10)
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.25, axis="y")
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

        plt.suptitle("Distribution of Successful Target Breaches Across Active Targets", fontsize=11.5, fontweight="bold")
        plt.tight_layout()
        plot4 = os.path.join(results_dir, "target_choice_distribution.png")
        plt.savefig(plot4, dpi=120)
        plt.close()
        print(f"Generated chart: {plot4}")

    # 5. Grouped Bar Chart: Archetype Breakdown Across K
    plt.figure(figsize=(10.5, 5.5))
    bar_archetypes = ["learned_defender", "pursuit", "patroller", "center_camper", "stay"]
    x = np.arange(len(bar_archetypes))
    width = 0.8 / len(k_values)

    palette = ["#3a86ff", "#8338ec", "#ff006e", "#fb5607"]

    for idx, k in enumerate(k_values):
        k_key = f"K_{k}"
        win_rates = []
        for arch in bar_archetypes:
            if k_key in all_results and arch in all_results[k_key]:
                win_rates.append(all_results[k_key][arch]["attacker_win_rate"] * 100.0)
            else:
                win_rates.append(0.0)
        offset = (idx - len(k_values) / 2.0 + 0.5) * width
        plt.bar(x + offset, win_rates, width, label=f"K = {k}", color=palette[idx % len(palette)], alpha=0.85)

    plt.title("Attacker Win Rate by Defender Archetype Across Target Counts K", fontsize=12, fontweight="bold")
    plt.xlabel("Defender Archetype", fontsize=11)
    plt.ylabel("Attacker Win Rate (%)", fontsize=11)
    plt.xticks(x, [a.replace("_", " ").title() for a in bar_archetypes], fontsize=10)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(loc="upper left", fontsize=9.5)
    plt.tight_layout()
    plot5 = os.path.join(results_dir, "win_rate_by_defender_archetype.png")
    plt.savefig(plot5, dpi=120)
    plt.close()
    print(f"Generated chart: {plot5}")


def main():
    parser = argparse.ArgumentParser(description="Scientific comparison of Single vs Multi-Target Arena.")
    parser.add_argument("--targets", nargs="+", type=int, default=[1, 2, 3, 5],
                        help="List of target counts K to benchmark")
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of paired test episodes per condition")
    parser.add_argument("--models-dir", type=str, default="models/multi_target",
                        help="Path to directory containing multi-target models")
    parser.add_argument("--phase4-model", type=str, default="models/phase4_attacker.pth",
                        help="Path to single-target baseline model")
    parser.add_argument("--results-dir", type=str, default="results/multi_target",
                        help="Directory to save comparison metrics and figures")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for paired trials")
    parser.add_argument("--replot-only", action="store_true",
                        help="Re-generate figures from existing results/multi_target/comparison_metrics.json")
    args = parser.parse_args()

    if args.replot_only:
        json_path = os.path.join(args.results_dir, "comparison_metrics.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            generate_comparison_figures(all_results, args.targets, args.results_dir)
            print(f"Re-generated all comparison figures in {args.results_dir} from {json_path}")
            return
        else:
            print(f"Metrics file {json_path} not found. Running full comparison instead.")

    run_comprehensive_comparison(
        k_values=args.targets,
        num_episodes=args.episodes,
        models_dir=args.models_dir,
        phase4_model_path=args.phase4_model,
        results_dir=args.results_dir,
        seed_base=args.seed,
    )


if __name__ == "__main__":
    main()
