import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_single_vs_multi_comparison_plot(
    single_target_json: str = "results/single_target/comparison_metrics.json",
    multi_target_json: str = "results/multi_target_k2/comparison_metrics.json",
    output_path: str = "results/multi_target_k2/k1_vs_k2_comparison.png",
):
    if not os.path.exists(multi_target_json) and os.path.exists("results/multi_target/comparison_metrics.json"):
        multi_target_json = "results/multi_target/comparison_metrics.json"

    if not os.path.exists(single_target_json) or not os.path.exists(multi_target_json):
        print(f"Missing input json files: {single_target_json} or {multi_target_json}")
        return

    with open(single_target_json, "r", encoding="utf-8") as f:
        single_data = json.load(f)
    with open(multi_target_json, "r", encoding="utf-8") as f:
        multi_raw = json.load(f)

    # Multi-target data for K=2
    multi_data = multi_raw.get("K_2", multi_raw)

    archetypes = [
        ("north_camper", "North Camper (Y=1)"),
        ("center_camper", "Center Camper (Y=3)"),
        ("south_camper", "South Camper (Y=5)"),
        ("patroller", "River Patroller"),
        ("stay", "Stationary Stay"),
        ("random", "Random Walk"),
        ("pursuit", "Greedy Pursuit"),
        ("learned_defender", "Learned Def"),
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

    for k, _ in archetypes:
        s_m = single_data.get(k, {})
        m_m = multi_data.get(k, {})

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

    # 1. Dark theme figure matching dashboard: k1_vs_k2_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), facecolor="#090d16")
    plt.subplots_adjust(hspace=0.38, wspace=0.22)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#0d131f")
            ax.tick_params(colors="#94a3b8")
            ax.grid(True, color=(1, 1, 1, 0.08), linestyle="--", axis="y")

    x = np.arange(len(archetypes))
    width = 0.36

    # Panel 1: Attacker Win Rate Comparison
    ax1 = axes[0, 0]
    ax1.bar(x - width/2, single_att_wins, width, label="Single-Target Citadel (K=1)", color="#475569", alpha=0.9)
    ax1.bar(x + width/2, multi_att_wins, width, label="Multi-Target Dual-Flank (K=2)", color="#ef4444", alpha=0.9)
    ax1.set_title("A. Attacker Win Rate (%) Comparison", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Attacker Win Rate (%)", color="#94a3b8", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, rotation=25, ha="right", color="#94a3b8", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper left", fontsize=8.5)

    # Panel 2: Defender Capture Rate Comparison
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, single_def_wins, width, label="Single-Target Citadel (K=1)", color="#475569", alpha=0.9)
    ax2.bar(x + width/2, multi_def_wins, width, label="Multi-Target Dual-Flank (K=2)", color="#38bdf8", alpha=0.9)
    ax2.set_title("B. Defender Capture Rate (%) Comparison", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Defender Capture Rate (%)", color="#94a3b8", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, rotation=25, ha="right", color="#94a3b8", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    # Panel 3: Absolute Delta in Percentage Points (Multi - Single)
    ax3 = axes[1, 0]
    colors_att = ["#10b981" if d >= 0 else "#f43f5e" for d in diff_att_pp]
    bars_diff = ax3.bar(x, diff_att_pp, width=0.55, color=colors_att, edgecolor="none", alpha=0.9)
    ax3.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.9, alpha=0.7)
    ax3.set_title("C. Net Attacker Win Rate Shift (Δ pp: K=2 - K=1)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax3.set_ylabel("Δ Attacker Win Rate (% points)", color="#94a3b8", fontsize=10.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_labels, rotation=25, ha="right", color="#94a3b8", fontsize=9)
    ax3.set_ylim(min(-40, min(diff_att_pp) - 10), max(60, max(diff_att_pp) + 12))
    for bar in bars_diff:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 1.5 if h >= 0 else -3.5
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + offset, f"{h:+.1f} pp", ha="center", va=va, color="#f8fafc", fontsize=8, fontweight="bold")

    # Panel 4: Episode Duration Comparison (Steps)
    ax4 = axes[1, 1]
    ax4.bar(x - width/2, single_steps, width, label="Single-Target Citadel (K=1)", color="#475569", alpha=0.9)
    ax4.bar(x + width/2, multi_steps, width, label="Multi-Target Dual-Flank (K=2)", color="#a855f7", alpha=0.9)
    ax4.set_title("D. Episode Duration (Steps) Comparison (T ≤ 100)", color="#f8fafc", fontsize=11.5, fontweight="bold")
    ax4.set_ylabel("Average Steps to Termination", color="#94a3b8", fontsize=10.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(arch_labels, rotation=25, ha="right", color="#94a3b8", fontsize=9)
    ax4.set_ylim(0, 105)
    ax4.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    plt.suptitle("Single-Target Citadel (K=1) vs Multi-Target Dual-Flank (K=2) Comparison\n500 Paired Test Episodes per Condition (T = 100 Horizon)", color="#f8fafc", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated comparison figure: {output_path}")

    # 2. k2_robustness_archetype.png for the Robustness page
    robustness_path = os.path.join(os.path.dirname(output_path), "k2_robustness_archetype.png")
    fig2, (rax1, rax2) = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor="#090d16")
    rax1.set_facecolor("#0d131f")
    rax2.set_facecolor("#0d131f")
    for rax in (rax1, rax2):
        rax.tick_params(colors="#94a3b8")
        rax.grid(True, color=(1, 1, 1, 0.08), linestyle="--", axis="y")

    rax1.bar(x - width/2, multi_att_wins, width, label="Attacker Win Rate (%)", color="#ef4444", alpha=0.9)
    rax1.bar(x + width/2, multi_def_wins, width, label="Defender Capture Rate (%)", color="#38bdf8", alpha=0.9)
    rax1.set_title("K=2 Multi-Target Robustness Across Defender Archetypes", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    rax1.set_xticks(x)
    rax1.set_xticklabels(arch_labels, rotation=35, ha="right", color="#94a3b8", fontsize=8.5)
    rax1.set_ylabel("Rate (%)", color="#94a3b8", fontsize=10)
    rax1.set_ylim(0, 105)
    rax1.legend(facecolor="#090d16", edgecolor=(1, 1, 1, 0.15), labelcolor="#f8fafc", loc="upper right", fontsize=8.5)

    bars_step = rax2.bar(x, multi_steps, width=0.55, color="#a855f7", alpha=0.88)
    rax2.set_title("K=2 Average Episode Steps to Termination (T ≤ 100)", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    rax2.set_xticks(x)
    rax2.set_xticklabels(arch_labels, rotation=35, ha="right", color="#94a3b8", fontsize=8.5)
    rax2.set_ylabel("Steps", color="#94a3b8", fontsize=10)
    rax2.set_ylim(0, 105)
    for bar in bars_step:
        h = bar.get_height()
        rax2.text(bar.get_x() + bar.get_width()/2.0, h + 1.5, f"{h:.1f}", ha="center", va="bottom", color="#f8fafc", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(robustness_path, dpi=130, facecolor=fig2.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Generated robustness figure: {robustness_path}")


if __name__ == "__main__":
    generate_single_vs_multi_comparison_plot()
