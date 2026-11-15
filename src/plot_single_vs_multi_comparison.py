import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

K_CONFIG_NAMES = {
    1: "Citadel Baseline (K=1)",
    2: "Dual-Flank Corridors (K=2)",
    3: "Bridge-Front Corridors (K=3)",
    5: "Full Eastern Frontier (K=5)",
}

def generate_single_vs_multi_comparison_plot(
    k: int = 2,
    single_target_json: str = "results/single_target/comparison_metrics.json",
    multi_target_json: str = None,
    output_path: str = None,
):
    if multi_target_json is None:
        cand_k = f"results/multi_target_k{k}/comparison_metrics.json"
        multi_target_json = cand_k if os.path.exists(cand_k) else "results/multi_target/comparison_metrics.json"

    if output_path is None:
        output_path = f"results/multi_target_k{k}/k1_vs_k{k}_comparison.png"

    if not os.path.exists(single_target_json) or not os.path.exists(multi_target_json):
        print(f"Missing input json files: {single_target_json} or {multi_target_json}")
        return

    with open(single_target_json, "r", encoding="utf-8") as f:
        single_data = json.load(f)
    with open(multi_target_json, "r", encoding="utf-8") as f:
        multi_raw = json.load(f)

    # Multi-target data for K
    k_key = f"K_{k}"
    multi_data = multi_raw.get(k_key, multi_raw)

    # Save isolated comparison metrics for this K if missing
    k_dir = f"results/multi_target_k{k}"
    os.makedirs(k_dir, exist_ok=True)
    isolated_json = os.path.join(k_dir, "comparison_metrics.json")
    if not os.path.exists(isolated_json) or k != 2:
        with open(isolated_json, "w", encoding="utf-8") as f:
            json.dump({k_key: multi_data}, f, indent=2)

    # 4 key tactical defender archetypes
    archetypes = [
        ("north_camper", "North Camper (Y=1)"),
        ("center_camper", "Center Camper (Y=3)"),
        ("south_camper", "South Camper (Y=5)"),
        ("patroller", "River Patroller"),
    ]

    arch_labels = [a[1] for a in archetypes]

    single_att_wins = []
    single_def_wins = []
    single_steps = []

    multi_att_wins = []
    multi_def_wins = []
    multi_steps = []

    diff_att_pp = []
    diff_def_pp = []
    diff_steps = []

    for arch_key, _ in archetypes:
        s_m = single_data.get(arch_key, {})
        m_m = multi_data.get(arch_key, {})

        s_att = s_m.get("attacker_win_rate", 0.0) * 100.0
        s_def = s_m.get("defender_win_rate", 0.0) * 100.0
        s_stp = s_m.get("avg_episode_steps", 0.0)

        m_att = m_m.get("attacker_win_rate", 0.0) * 100.0
        m_def = m_m.get("defender_win_rate", 0.0) * 100.0
        m_stp = m_m.get("avg_episode_steps", 0.0)

        single_att_wins.append(s_att)
        single_def_wins.append(s_def)
        single_steps.append(s_stp)

        multi_att_wins.append(m_att)
        multi_def_wins.append(m_def)
        multi_steps.append(m_stp)

        diff_att_pp.append(m_att - s_att)
        diff_def_pp.append(m_def - s_def)
        diff_steps.append(m_stp - s_stp)

    k_label = K_CONFIG_NAMES.get(k, f"Multi-Target (K={k})")

    # 1. Dark theme figure: Top 2 graphs (Attacker Win Rate & Defender Capture Rate)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), facecolor="#090d16")
    plt.subplots_adjust(wspace=0.22, top=0.82, bottom=0.22)

    for ax in (ax1, ax2):
        ax.set_facecolor("#0d131f")
        ax.tick_params(colors="#94a3b8")
        ax.grid(True, color=(1, 1, 1, 0.08), linestyle="--", axis="y")

    x = np.arange(len(archetypes))
    width = 0.36

    # Panel 1: Attacker Win Rate Comparison
    b1 = ax1.bar(x - width/2, single_att_wins, width, label="Single-Target Citadel (K=1)", color="#475569", alpha=0.9)
    b2 = ax1.bar(x + width/2, multi_att_wins, width, label=f"Multi-Target {k_label}", color="#ef4444", alpha=0.9)
    ax1.set_title(f"A. Attacker Win Rate (%) Comparison: K=1 vs K={k}", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Attacker Win Rate (%)", color="#94a3b8", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, rotation=15, ha="right", color="#94a3b8", fontsize=10)
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper left", fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#f8fafc", fontsize=8, fontweight="bold")

    # Panel 2: Defender Capture Rate Comparison
    b3 = ax2.bar(x - width/2, single_def_wins, width, label="Single-Target Citadel (K=1)", color="#475569", alpha=0.9)
    b4 = ax2.bar(x + width/2, multi_def_wins, width, label=f"Multi-Target {k_label}", color="#38bdf8", alpha=0.9)
    ax2.set_title(f"B. Defender Capture Rate (%) Comparison: K=1 vs K={k}", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Defender Capture Rate (%)", color="#94a3b8", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, rotation=15, ha="right", color="#94a3b8", fontsize=10)
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)
    for bar in b4:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#f8fafc", fontsize=8, fontweight="bold")

    plt.suptitle(f"Single-Target Citadel (K=1) vs Multi-Target {k_label} Comparison\n500 Paired Test Episodes per Condition (T = 100 Horizon)", color="#f8fafc", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated comparison figure: {output_path}")

    # 2. Robustness figure
    robustness_path = os.path.join(os.path.dirname(output_path), f"k{k}_robustness_archetype.png")
    fig2, (rax1, rax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor="#090d16")
    rax1.set_facecolor("#0d131f")
    rax2.set_facecolor("#0d131f")
    for rax in (rax1, rax2):
        rax.tick_params(colors="#94a3b8")
        rax.grid(True, color=(1, 1, 1, 0.08), linestyle="--", axis="y")

    rax1.bar(x - width/2, multi_att_wins, width, label="Attacker Win Rate (%)", color="#ef4444", alpha=0.9)
    rax1.bar(x + width/2, multi_def_wins, width, label="Defender Capture Rate (%)", color="#38bdf8", alpha=0.9)
    rax1.set_title(f"K={k} Multi-Target Robustness Across Defender Archetypes", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    rax1.set_xticks(x)
    rax1.set_xticklabels(arch_labels, rotation=20, ha="right", color="#94a3b8", fontsize=9.5)
    rax1.set_ylabel("Rate (%)", color="#94a3b8", fontsize=10)
    rax1.set_ylim(0, 105)
    rax1.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    bars_step = rax2.bar(x, multi_steps, width=0.55, color="#a855f7", alpha=0.88)
    rax2.set_title(f"K={k} Average Episode Steps to Termination (T ≤ 100)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    rax2.set_xticks(x)
    rax2.set_xticklabels(arch_labels, rotation=20, ha="right", color="#94a3b8", fontsize=9.5)
    rax2.set_ylabel("Steps", color="#94a3b8", fontsize=10)
    rax2.set_ylim(0, 105)
    for bar in bars_step:
        h = bar.get_height()
        rax2.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}", ha="center", va="bottom", color="#f8fafc", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(robustness_path, dpi=130, facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated robustness figure: {robustness_path}")


def generate_all_k_comparison_plots():
    for k in [2, 3, 5]:
        generate_single_vs_multi_comparison_plot(k=k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K vs Single Target Comparison Figures.")
    parser.add_argument("--k", type=int, default=None, help="Target count K (2, 3, 5) or None for all")
    args = parser.parse_args()

    if args.k is not None:
        generate_single_vs_multi_comparison_plot(k=args.k)
    else:
        generate_all_k_comparison_plots()
