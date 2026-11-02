import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_single_vs_multi_comparison_plot(
    single_target_json: str = "results/single_target/comparison_metrics.json",
    multi_target_json: str = "results/multi_target/comparison_metrics.json",
    output_path: str = "results/single_vs_multi_target_comparison.png",
):
    if not os.path.exists(single_target_json) or not os.path.exists(multi_target_json):
        print(f"Missing input json files: {single_target_json} or {multi_target_json}")
        return

    with open(single_target_json, "r", encoding="utf-8") as f:
        single_data = json.load(f)
    with open(multi_target_json, "r", encoding="utf-8") as f:
        multi_data = json.load(f)

    archetypes = [
        ("learned_defender", "Learned Def"),
        ("pursuit", "Pursuit"),
        ("patroller", "Patroller"),
        ("center_camper", "Center Camper"),
        ("north_camper", "North Camper"),
        ("south_camper", "South Camper"),
        ("stay", "Stay"),
        ("random", "Random"),
    ]

    arch_keys = [a[0] for a in archetypes]
    arch_labels = [a[1] for a in archetypes]

    single_att_wins = []
    single_def_wins = []
    single_steps = []

    multi5_att_wins = []
    multi5_def_wins = []
    multi5_steps = []

    diff_att_pp = []
    diff_def_pp = []
    diff_steps = []

    for k, _ in archetypes:
        s_m = single_data.get(k, {})
        m5_m = multi_data.get("K_5", {}).get(k, {})

        s_att = s_m.get("attacker_win_rate", 0.0) * 100.0
        s_def = s_m.get("defender_win_rate", 0.0) * 100.0
        s_stp = s_m.get("avg_episode_steps", 0.0)

        m_att = m5_m.get("attacker_win_rate", 0.0) * 100.0
        m_def = m5_m.get("defender_win_rate", 0.0) * 100.0
        m_stp = m5_m.get("avg_episode_steps", 0.0)

        single_att_wins.append(s_att)
        single_def_wins.append(s_def)
        single_steps.append(s_stp)

        multi5_att_wins.append(m_att)
        multi5_def_wins.append(m_def)
        multi5_steps.append(m_stp)

        diff_att_pp.append(m_att - s_att)
        diff_def_pp.append(m_def - s_def)
        diff_steps.append(m_stp - s_stp)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plt.subplots_adjust(hspace=0.35, wspace=0.22)

    x = np.arange(len(archetypes))
    width = 0.36

    # Panel 1: Attacker Win Rate Comparison
    ax1 = axes[0, 0]
    b1 = ax1.bar(x - width/2, single_att_wins, width, label="Original Single-Target (Citadel)", color="#457b9d", alpha=0.9)
    b2 = ax1.bar(x + width/2, multi5_att_wins, width, label="New Multi-Target (K=5 Frontier)", color="#e63946", alpha=0.9)
    ax1.set_title("A. Attacker Win Rate (%) Comparison", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Attacker Win Rate (%)", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.25, axis="y")
    ax1.legend(loc="upper left", fontsize=8.5)

    # Panel 2: Defender Capture Rate Comparison
    ax2 = axes[0, 1]
    b3 = ax2.bar(x - width/2, single_def_wins, width, label="Original Single-Target (Citadel)", color="#457b9d", alpha=0.9)
    b4 = ax2.bar(x + width/2, multi5_def_wins, width, label="New Multi-Target (K=5 Frontier)", color="#1d3557", alpha=0.9)
    ax2.set_title("B. Defender Capture Rate (%) Comparison", fontsize=11.5, fontweight="bold")
    ax2.set_ylabel("Defender Capture Rate (%)", fontsize=10.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.25, axis="y")
    ax2.legend(loc="upper right", fontsize=8.5)

    # Panel 3: Absolute Delta in Percentage Points (Multi - Single)
    ax3 = axes[1, 0]
    colors_att = ["#2a9d8f" if d >= 0 else "#e76f51" for d in diff_att_pp]
    bars_diff = ax3.bar(x, diff_att_pp, width=0.55, color=colors_att, edgecolor="#333333", alpha=0.88)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.9, alpha=0.7)
    ax3.set_title("C. Net Attacker Win Rate Shift (Δ Percentage Points: Multi - Single)", fontsize=11.5, fontweight="bold")
    ax3.set_ylabel("Δ Attacker Win Rate (% points)", fontsize=10.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=9)
    ax3.set_ylim(-35, 65)
    ax3.grid(True, alpha=0.25, axis="y")
    for bar in bars_diff:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 1.5 if h >= 0 else -3.5
        ax3.text(bar.get_x() + bar.get_width()/2.0, h + offset, f"{h:+.1f} pp", ha="center", va=va, fontsize=8.5, fontweight="bold")

    # Panel 4: Episode Duration Comparison (Steps)
    ax4 = axes[1, 1]
    b5 = ax4.bar(x - width/2, single_steps, width, label="Original Single-Target (Citadel)", color="#a8dadc", edgecolor="#457b9d", alpha=0.95)
    b6 = ax4.bar(x + width/2, multi5_steps, width, label="New Multi-Target (K=5 Frontier)", color="#f4a261", edgecolor="#e76f51", alpha=0.95)
    ax4.set_title("D. Episode Duration (Steps) Comparison", fontsize=11.5, fontweight="bold")
    ax4.set_ylabel("Average Steps to Termination", fontsize=10.5)
    ax4.set_xticks(x)
    ax4.set_xticklabels(arch_labels, rotation=25, ha="right", fontsize=9)
    ax4.grid(True, alpha=0.25, axis="y")
    ax4.legend(loc="upper right", fontsize=8.5)

    plt.suptitle("Original Single-Target System vs New Multi-Target System (K=5) Comparison\n500 Paired Test Episodes per Condition (Fixed Seeds 42–541)", fontsize=13, fontweight="bold")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Generated comparison figure: {output_path}")

if __name__ == "__main__":
    generate_single_vs_multi_comparison_plot()
