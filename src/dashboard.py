import streamlit as st
import json
import os
import sys
import streamlit.components.v1 as components
import pandas as pd

# Ensure local imports work whether executed from root or src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from multi_target_environment import MultiTargetArenaEnv, PRESET_TARGETS, STATE_DIM, MAX_TARGETS
except ImportError:
    MultiTargetArenaEnv = None
    PRESET_TARGETS = {
        1: [(10, 3)],
        2: [(10, 1), (10, 5)],
        3: [(10, 1), (10, 3), (10, 5)],
        5: [(10, 0), (10, 1), (10, 3), (10, 5), (10, 6)],
    }
    STATE_DIM = 19
    MAX_TARGETS = 5

st.set_page_config(
    layout="wide",
    page_title="Red Queen MARL Arena",
    page_icon="👑",
    initial_sidebar_state="collapsed"
)

# Impeccable Craft Styling: Ultra-clean executive dark theme, zero radio circles, no awkward wrapping
st.markdown("""
<style>
    /* 1. Reset standard Streamlit chrome */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"], 
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1360px !important;
        margin: 0 auto !important;
    }
    
    /* 2. Unified Header Masthead */
    .masthead-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 10px 18px;
        margin-bottom: 10px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    
    .masthead-left {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .masthead-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 8px;
        font-size: 1.15rem;
    }
    
    .masthead-title {
        font-size: 0.96rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    
    .masthead-title span {
        color: #38bdf8;
        font-weight: 600;
    }
    
    .masthead-sub {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    
    .masthead-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 2px;
    }
    
    .masthead-affiliation {
        font-size: 0.76rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    
    .masthead-authors {
        font-size: 0.72rem;
        color: #64748b;
    }
    
    .masthead-authors strong {
        color: #94a3b8;
        font-weight: 600;
    }

    /* 3. Segmented Control Overrides: Pure Elegance */
    div[data-testid="stSegmentedControl"] {
        position: sticky !important;
        top: 0px !important;
        z-index: 99999 !important;
        background: rgba(9, 13, 22, 0.95) !important;
        backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 18px !important;
    }
    
    div[data-testid="stSegmentedControl"] > div {
        display: flex !important;
        width: 100% !important;
        gap: 4px !important;
    }
    
    /* Segment buttons */
    div[data-testid="stSegmentedControl"] button {
        flex: 1 !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        border-radius: 7px !important;
        padding: 6px 10px !important;
        font-size: 0.80rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        white-space: nowrap !important;
        transition: all 0.16s ease-in-out !important;
        cursor: pointer !important;
        text-align: center !important;
        outline: none !important;
    }
    
    /* Tactical Grid & Status Badges */
    .tactical-grid {
        border-collapse: separate;
        border-spacing: 4px;
        margin: 10px auto;
    }
    .tactical-cell {
        width: 44px;
        height: 38px;
        text-align: center;
        vertical-align: middle;
        font-size: 0.70rem;
        font-family: monospace;
        font-weight: 700;
        border-radius: 6px;
    }
    .status-badge-ready {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #34d399;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .status-badge-missing {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
        font-size: 0.74rem;
        font-weight: 600;
    }
    
    div[data-testid="stSegmentedControl"] button:hover {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #f8fafc !important;
    }
    
    /* Active Segment */
    div[data-testid="stSegmentedControl"] button[aria-selected="true"],
    div[data-testid="stSegmentedControl"] button[data-selected="true"] {
        background: rgba(56, 189, 248, 0.16) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.25) !important;
    }

    /* Fallback in case st.radio is rendered */
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] svg,
    div[data-testid="stRadio"] [data-baseweb="radio"] div {
        display: none !important;
    }

    /* 4. Stat & Metric Cards (Impeccable Visual Floor) */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: #090d16;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 2px solid #38bdf8;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: left;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.55rem;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    
    .metric-subtitle {
        font-size: 0.74rem;
        color: #38bdf8;
        font-weight: 500;
        margin-top: 4px;
    }
    
    /* 5. Center & constrain charts */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin: 12px 0 20px 0 !important;
    }
    
    div[data-testid="stImage"] > img {
        max-width: 820px !important;
        max-height: 480px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.6) !important;
    }
</style>
""", unsafe_allow_html=True)

# Unified Executive Masthead (Clean, Single Line, No Clutter)
st.markdown("""
<div class="masthead-card">
    <div class="masthead-left">
        <div class="masthead-logo">👑</div>
        <div>
            <div class="masthead-title">Red Queen Arena <span>| Autonomous Multi-Agent RL</span></div>
            <div class="masthead-sub">Three Bridges Bottleneck Game • Empirical Game-Theoretic Analysis (EGTA)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =============================================================
# MODE SWITCHER — Single Target vs Multi-Target K=2
# =============================================================
if "dashboard_mode" not in st.session_state:
    st.session_state.dashboard_mode = "single"

mc1, mc2 = st.columns(2)
with mc1:
    if st.button(
        "⚔️ Single Target (Citadel)",
        use_container_width=True,
        type="primary" if st.session_state.dashboard_mode == "single" else "secondary",
        key="btn_mode_single"
    ):
        st.session_state.dashboard_mode = "single"
        if "selected_page" in st.session_state:
            del st.session_state["selected_page"]
        st.rerun()
with mc2:
    if st.button(
        "🎯 Multi-Target (K=2 Flank Perimeter)",
        use_container_width=True,
        type="primary" if st.session_state.dashboard_mode == "multi_k2" else "secondary",
        key="btn_mode_multi"
    ):
        st.session_state.dashboard_mode = "multi_k2"
        if "selected_page" in st.session_state:
            del st.session_state["selected_page"]
        st.rerun()

# Per-mode navigation pages
if st.session_state.dashboard_mode == "single":
    PAGES = [
        "🎮 3D Arena",
        "📜 Game Theory",
        "🎯 Baseline PPO",
        "🔄 Naive Self-Play",
        "🛡️ Dual-Archive",
        "📊 Robustness (EGTA)",
        "🌍 Applications"
    ]
else:
    PAGES = [
        "🎮 3D Arena (K=2)",
        "🎯 Fixed Baseline",
        "🔄 Red Queen Self-Play",
        "🛡️ Dual-Archive K=2",
        "📊 Robustness (K=2)",
        "⚖️ vs Single Target"
    ]

# Reset page selection when mode changes
if "selected_page" not in st.session_state or st.session_state.get("_last_mode") != st.session_state.dashboard_mode:
    st.session_state.selected_page = PAGES[0]
    st.session_state._last_mode = st.session_state.dashboard_mode

selected_page = st.segmented_control(
    "Navigation",
    PAGES,
    default=st.session_state.selected_page,
    selection_mode="single",
    label_visibility="collapsed",
    width="stretch"
)
if not selected_page:
    selected_page = PAGES[0]
st.session_state.selected_page = selected_page


def display_image(image_path):
    if os.path.exists(image_path):
        col_l, col_m, col_r = st.columns([1.5, 7, 1.5])
        with col_m:
            st.image(image_path, width="stretch")
    else:
        st.warning(f"Artifact not found: {image_path}. Please execute the corresponding evaluation or training script.")

def get_all_replays_data():
    """Load and safely encode trajectory JSONs for WebGL injection."""
    def load_json_safe(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.dumps(json.load(f))
                except json.JSONDecodeError:
                    return "null"
        return "null"
    return (
        load_json_safe("replays/fixed_opponent.json"),
        load_json_safe("replays/red_queen.json"),
        load_json_safe("replays/red_queen_history.json")
    )

def get_k2_replays_data():
    """Load and safely encode K=2 trajectory JSONs for WebGL injection."""
    def load_json_safe(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.dumps(json.load(f))
                except json.JSONDecodeError:
                    return "null"
        return "null"
    return (
        load_json_safe("replays/multi_target_k2/k2_fixed_opponent.json"),
        load_json_safe("replays/multi_target_k2/k2_red_queen.json"),
        load_json_safe("replays/multi_target_k2/k2_dual_archive.json")
    )


def load_json_safe_file(filepath: str):
    """Safely read and parse a JSON file, returning None if missing or empty."""
    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def format_percentage(val, ci=None):
    if val is None:
        return "N/A"
    pct = val * 100.0
    if ci is not None:
        return f"{pct:.1f}% ± {ci * 100.0:.1f}%"
    return f"{pct:.1f}%"

def format_steps(val, sem=None):
    if val is None:
        return "N/A"
    if sem is not None:
        return f"{val:.2f} ± {sem:.2f}"
    return f"{val:.2f}"

def render_tactical_grid_html(target_coords):
    target_set = {tuple(t): idx for idx, t in enumerate(target_coords)}
    river_obstacles = {(5, 0), (5, 2), (5, 4), (5, 6)}
    bridges = {(5, 1): "North", (5, 3): "Center", (5, 5): "South"}
    
    html = ['<div style="overflow-x:auto;"><table class="tactical-grid">']
    html.append('<tr><th style="color:#64748b;font-size:0.68rem;padding:3px;">Y \\ X</th>')
    for x in range(11):
        html.append(f'<th style="color:#64748b;font-size:0.68rem;width:44px;text-align:center;">{x}</th>')
    html.append('</tr>')
    
    for y in range(7):
        html.append(f'<tr><td style="color:#64748b;font-size:0.68rem;text-align:center;font-weight:600;">{y}</td>')
        for x in range(11):
            pos = (x, y)
            if pos in target_set:
                t_idx = target_set[pos]
                bg = 'rgba(239, 68, 68, 0.28)'
                bdr = '#ef4444'
                clr = '#fca5a5'
                content = f'🎯 T{t_idx}'
            elif pos in river_obstacles:
                bg = 'rgba(14, 165, 233, 0.18)'
                bdr = 'rgba(14, 165, 233, 0.35)'
                clr = '#38bdf8'
                content = '🌊'
            elif pos in bridges:
                b_name = bridges[pos][0]
                bg = 'rgba(2, 132, 199, 0.38)'
                bdr = '#0284c7'
                clr = '#bae6fd'
                content = f'🌉 {b_name}'
            elif x <= 1:
                bg = 'rgba(16, 185, 129, 0.08)'
                bdr = 'rgba(16, 185, 129, 0.22)'
                clr = '#6ee7b7'
                content = '⚔️ Att' if (x == 0 and y == 3) else ''
            elif x in (7, 8):
                bg = 'rgba(99, 102, 241, 0.08)'
                bdr = 'rgba(99, 102, 241, 0.22)'
                clr = '#a5b4fc'
                content = '🛡️ Def' if (x == 7 and y == 3) else ''
            else:
                bg = 'rgba(15, 23, 42, 0.55)'
                bdr = 'rgba(255, 255, 255, 0.05)'
                clr = '#475569'
                content = ''
            html.append(f'<td style="background:{bg};border:1px solid {bdr};color:{clr};" class="tactical-cell">{content}</td>')
        html.append('</tr>')
    html.append('</table></div>')
    return ''.join(html)


# =============================================================
# PAGE ROUTING
# =============================================================

if st.session_state.dashboard_mode == "single":
    # =========================================================
    # SINGLE TARGET MODE — Original 7 pages, untouched
    # =========================================================

    if selected_page == "🎮 3D Arena":
        html_path = "dashboard_3d.html"
        if os.path.exists(html_path):
            fixed_json, rq_json, da_json = get_all_replays_data()
            with open(html_path, "r", encoding="utf-8") as hf:
                html_raw = hf.read()

            injection_script = f"""<script>
            window.ALL_REPLAYS = {{
                'fixed': {fixed_json},
                'redqueen': {rq_json},
                'doublearchive': {da_json}
            }};
            </script>"""
            injected = html_raw.replace("/* __EMBEDDED_REPLAY_DATA__ */", injection_script)
            components.html(injected, height=750, scrolling=False)
        else:
            st.warning("3D Arena file not found.")

    elif selected_page == "📜 Game Theory":
        if os.path.exists("docs/phase1_game_theory_spec.md"):
            with open("docs/phase1_game_theory_spec.md", "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.warning("Specification document not found.")

    elif selected_page == "🎯 Baseline PPO":
        st.title("Fixed Opponent Baseline (PPO)")
        st.markdown("""
        In this setup, the Defender remains stationary (camping) to benchmark single-agent PPO pathfinding across the $11 \\times 7$ Three Bridges topology.
        
        Using discounted potential shaping, the Attacker rapidly learns collision-free navigation and decisive river crossing without getting trapped in obstacle chokepoints.
        """)
        display_image("docs/phase3_training_curve.png")

    elif selected_page == "🔄 Naive Self-Play":
        st.title("Naive Self-Play & The Red Queen Dynamic")
        st.markdown(r"""
        Both agents train simultaneously via Proximal Policy Optimization (PPO) without historical memory. Forced to navigate through the three chokepoints, their policies cycle in response to immediate adversary adaptations.

        **The Pathology of Catastrophic Forgetting:**  
        When the Defender specializes in blocking the North bridge, the Attacker shifts to the South corridor. In doing so, backpropagation overwrites policy weights, forgetting how to breach Center or North. When the Defender responds by shifting South, the cycle repeats.
        
        **Strict Zero-Sum Invariant:**  
        The interaction satisfies $R_{\text{Attacker}} + R_{\text{Defender}} \equiv 0.000000$ identically at every timestep. Potential-based shaping $F(s, s')$ is applied symmetrically ($+F$ to Attacker, $-F$ to Defender) with discount $\gamma = 0.99$, guaranteeing policy invariance under Ng et al. (1999) and Devlin & Kudenko (2011).
        """)
        display_image("docs/phase4_marl_curve.png")

    elif selected_page == "🛡️ Dual-Archive":
        st.title("Dual-Archive Co-Evolution (Mitigating Route Collapse)")
        st.markdown(r"""
        To anchor learning against policy cycling, both agents maintain persistent historical snapshot reservoirs ($\mathcal{H}_A, \mathcal{H}_D$), freezing checkpoints every 100 episodes after a 300-episode warmup.

        **Symmetric Opponent Mixture:**
        - **60% Active Self-Play:** Current Attacker vs Current Defender (both policies actively train).
        - **20% Attacker Historical Regularization:** Current Attacker vs Sampled Archived Defender (Attacker learns against past defenses).
        - **20% Defender Historical Regularization:** Sampled Archived Attacker vs Current Defender (Defender learns against past attacks).
        """)

        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Shannon Route Entropy</div>
                <div class="metric-value">1.567 bits</div>
                <div class="metric-subtitle">98.9% of Theoretical Maximum</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Navigation Timeouts</div>
                <div class="metric-value">0.0%</div>
                <div class="metric-subtitle">vs 17.3% in Naive Self-Play</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Corridor Traversal</div>
                <div class="metric-value">Balanced</div>
                <div class="metric-subtitle">26.7% N | 34.0% C | 39.3% S</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Convergence Parity</div>
                <div class="metric-value">Decisive</div>
                <div class="metric-subtitle">~65% Attacker | ~33% Defender</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        display_image("docs/phase6_archive_curve.png")

        st.markdown("---")
        st.subheader("🔬 Empirical Bridge-Route Diversity & Entropy Analysis")
        st.markdown(r"""
        Catastrophic route collapse is quantified using **Shannon Route Entropy** across the three bridge corridors over 150 paired evaluation seeds against an uncontested baseline:
        - **Theoretical Maximum Entropy ($H_{\max}$):** $\log_2(3) \approx 1.585$ bits (ideal $33.3\%$ uniform mixing).
        - **Phase 4 (Naive Self-Play):** Suffers asymmetric route collapse ($H = 1.470$ bits, $92.8\%$) with $46.0\%$ North bias, $16.1\%$ South neglect, and **$17.3\%$ unforced navigation timeouts**.
        - **Phase 6 (Dual Archive):** Restores strategic route diversity to **$1.567$ bits ($98.9\%$ of theoretical maximum)** with uniform multi-corridor coverage and **$0.0\%$ timeouts**.
        """)
        display_image("docs/bridge_diversity_chart.png")

    elif selected_page == "📊 Robustness (EGTA)":
        st.title("Empirical Game-Theoretic Analysis (EGTA)")
        st.markdown("""
        To evaluate strategic robustness without heuristic win-rate bias, trained models were evaluated against an archetypal adversary suite (North Camper, Center Camper, South Camper, River Patroller, and Learned Co-Evolved Defenders) across 150 paired pseudo-random seeds.
        """)

        st.markdown("""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Empirical Exploitability (δ)</div>
                <div class="metric-value">0.140</div>
                <div class="metric-subtitle">12.5% reduction vs Phase 4 (0.160)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Worst-Case Security Floor</div>
                <div class="metric-value">-0.187</div>
                <div class="metric-subtitle">Elevated from -0.207 in Phase 4</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Patroller Win Rate</div>
                <div class="metric-value">81.3%</div>
                <div class="metric-subtitle">vs 76.0% in Phase 4</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Pooled Paired Difference</div>
                <div class="metric-value">Δ = -0.6%</div>
                <div class="metric-subtitle">p = 0.771 (raw win rate invariant)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        display_image("docs/robustness_chart.png")

        st.markdown("---")
        st.subheader("📐 Minimax Exploitability & Security Analysis")
        st.markdown(r"""
        Using Empirical Game-Theoretic Analysis (EGTA) with Fictitious Play ($v^* = -0.046$):
        - **Phase 4 (Naive Self-Play):** Security level $v_{\text{sec}} = -0.207$, resulting in an exploitability score of **$\delta = 0.160$** (worst-case win rate $39.3\%$, patroller win rate $76.0\%$, mean margin $+0.190$).
        - **Phase 6 (Dual Archive):** Security level elevated to $v_{\text{sec}} = -0.187$, cutting exploitability to **$\delta = 0.140$** (a **12.5% reduction** in vulnerability, worst-case win rate $31.3\%$ against Phase 6 Defender, patroller win rate $81.3\%$, mean margin $+0.184$).
        - **Key Scientific Finding:** The pooled win-rate difference across all seeds is $\Delta = -0.6\%$ ($p = 0.771$, not statistically significant). This proves that **raw win rate is uninformative in competitive MARL**, whereas Shannon route entropy and empirical minimax exploitability isolate genuine strategic robustness.
        """)
        display_image("docs/nash_exploitability_chart.png")

    elif selected_page == "🌍 Applications":
        st.title("Domain-Agnostic Game Theory & Real-World Impact")
        st.markdown("""
        The Three Bridges topology serves as an abstract model for spatial and cyber bottleneck interactions where defensive pure strategies cannot simultaneously cover all infiltration paths.
        """)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💻 Cybersecurity: Adaptive Firewalls vs Autonomous Threats")
            st.markdown("""
            * **Attacker:** Self-mutating malware, botnets, or zero-day exploits.
            * **Defender:** Autonomous intrusion detection systems (IDS) / AI firewalls.
            * **Chokepoints:** Open network ports (HTTP 80, HTTPS 443, SSH 22).
            * **Citadel:** Secured internal database holding critical credentials.
            * **The Solution:** Dual-archive training forces the firewall to retain signatures against all historical vectors, preventing cyclical re-exploitation of unmonitored ports.
            """)

        with col2:
            st.subheader("🧬 Oncology & Medicine: Bacterial / Tumor Resistance")
            st.markdown("""
            * **Attacker:** Mutating pathogen, antibiotic-resistant bacteria, or tumor cells.
            * **Defender:** Targeted pharmaceutical drugs or combination immunotherapy.
            * **Chokepoints:** Cell surface receptors and metabolic signaling pathways.
            * **Citadel:** Host organ integrity and vital cell replication machinery.
            * **The Solution:** Analogous to combination therapy: training against historical drug combinations closes evolutionary escape corridors.
            """)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("🚗 Autonomous Driving: Multi-Lane Defensive Navigation")
            st.markdown("""
            * **Attacker:** Autonomous vehicle executing required off-ramp maneuvers.
            * **Defender:** Adversarial or erratic surrounding traffic.
            * **Chokepoints:** Distinct highway lanes during congested merges.
            * **Citadel:** Destination off-ramp.
            * **The Solution:** Historical archetype reservoirs expose the vehicle to diverse driving behaviors, avoiding overfitting to polite traffic.
            """)

        with col4:
            st.subheader("📈 Quantitative Finance: Market Makers vs Arbitrageurs")
            st.markdown("""
            * **Attacker:** High-frequency algorithmic arbitrageurs.
            * **Defender:** Automated market makers (AMMs) maintaining order liquidity.
            * **Chokepoints:** Order books and decentralized liquidity pools.
            * **Citadel:** Total risk-free alpha extraction.
            * **The Solution:** Training against an archive of historical volatility regimes prevents liquidity collapse during abrupt macroeconomic shifts.
            """)


elif st.session_state.dashboard_mode == "multi_k2":
    # =========================================================
    # MULTI-TARGET K=2 MODE — Separate 5-page sub-dashboard
    # =========================================================

    if selected_page == "🎮 3D Arena (K=2)":
        html_path = "dashboard_3d_k2.html"
        if os.path.exists(html_path):
            k2_fixed_json, k2_rq_json, k2_da_json = get_k2_replays_data()
            with open(html_path, "r", encoding="utf-8") as hf:
                html_raw = hf.read()

            injection_script = f"""<script>
            window.ALL_REPLAYS = {{
                'fixed': {k2_fixed_json},
                'redqueen': {k2_rq_json},
                'doublearchive': {k2_da_json}
            }};
            </script>"""
            injected = html_raw.replace("/* __EMBEDDED_REPLAY_DATA__ */", injection_script)
            components.html(injected, height=750, scrolling=False)
        else:
            st.warning("K=2 3D Arena file not found (dashboard_3d_k2.html).")
            st.code("The file should be at: dashboard_3d_k2.html (project root)", language="text")

    elif selected_page == "🎯 Fixed Baseline":
        st.title("Fixed Opponent Baseline — K=2 Dual-Flank Arena")
        st.markdown(r"""
        Benchmark evaluation of single-agent PPO pathfinding in the **K=2 Flank Perimeter** environment.
        The Defender remains strictly stationary (camping at spawn, $X \in [6, 9]$) to isolate Attacker navigation,
        chokepoint bridge choice, and dual-target selection without active adversary pursuit.
        """)

        # Load metrics for stay
        k2_metrics = load_json_safe_file("results/multi_target_k2/comparison_metrics.json")
        k2_stay = {}
        if k2_metrics and "K_2" in k2_metrics:
            k2_stay = k2_metrics["K_2"].get("stay", {})

        att_wr = k2_stay.get("attacker_win_rate")
        def_wr = k2_stay.get("defender_win_rate")
        timeout_r = k2_stay.get("timeout_rate")
        avg_steps = k2_stay.get("avg_episode_steps")

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Attacker Win Rate</div>
                <div class="metric-value">{format_percentage(att_wr)}</div>
                <div class="metric-subtitle">vs Stationary Defender (STAY)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Defender Capture Rate</div>
                <div class="metric-value">{format_percentage(def_wr)}</div>
                <div class="metric-subtitle">Accidental Spawn Intercept</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Timeout Rate</div>
                <div class="metric-value">{format_percentage(timeout_r)}</div>
                <div class="metric-subtitle">Zero Standoff Failure (T < 75)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Episode Length</div>
                <div class="metric-value">{format_steps(avg_steps)} steps</div>
                <div class="metric-subtitle">Optimal Pathfinding Trajectory</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 K=2 Fixed Baseline Learning Curve")
        display_image("results/multi_target_k2/fixed_baseline_k2_curve.png")
        st.markdown("""
        **K=2 Fixed Baseline Observations:**
        - **Rapid Pathfinding Convergence:** Attacker win rate climbs from 20% to >85% within 200 episodes as optimal bridge-routing is learned.
        - **Zero Standoff Timeouts (0%):** Unlike self-play standoff cycles, the Attacker achieves decisive breach trajectories with zero timeout stall.
        - **Balanced Dual-Flank Infiltration:** Attacker flexibly targets whichever flank corridor ($(10, 1)$ or $(10, 5)$) offers unobstructed passage relative to its Western spawn coordinates.
        """)

    elif selected_page in ["🔄 Red Queen Self-Play", "📈 Self-Play Training"]:
        st.title("K=2 Dual-Flank Self-Play Training (Phase 4 Equivalent)")
        st.markdown(r"""
        PPO self-play training for the **K=2 Flank Perimeter** configuration:
        two simultaneous target attractors at $(10,1)$ (North Flank) and $(10,5)$ (South Flank).

        The Defender faces a spatial dilemma — anchoring either flank leaves the opposite vulnerable.
        Both agents optimize decentralized actor-critic networks with PPO ($\epsilon=0.2$, $\gamma=0.99$) over 1500 episodes.
        """)

        # K=2 metric cards from comparison_metrics
        k2_metrics = load_json_safe_file("results/multi_target_k2/comparison_metrics.json")
        k2_learned = {}
        if k2_metrics and "K_2" in k2_metrics:
            k2_learned = k2_metrics["K_2"].get("learned_defender", {})

        att_wr = k2_learned.get("attacker_win_rate")
        def_wr = k2_learned.get("defender_win_rate")
        timeout_r = k2_learned.get("timeout_rate")
        avg_steps = k2_learned.get("avg_episode_steps")

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">K=2 Attacker Win Rate</div>
                <div class="metric-value">{format_percentage(att_wr)}</div>
                <div class="metric-subtitle">vs Learned PPO Defender</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Defender Capture Rate</div>
                <div class="metric-value">{format_percentage(def_wr)}</div>
                <div class="metric-subtitle">Dual-Flank Coverage Pressure</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Timeout Rate</div>
                <div class="metric-value">{format_percentage(timeout_r)}</div>
                <div class="metric-subtitle">Standoff Cycles (T ≥ 75)</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Episode Length</div>
                <div class="metric-value">{format_steps(avg_steps)} steps</div>
                <div class="metric-subtitle">K=2 Tactical Engagement</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📈 K=2 Self-Play Learning Curve")
        display_image("results/multi_target_k2/learning_curve_k2.png")
        st.markdown("""
        **K=2 Training Observations:**
        - Elevated timeout rates (50–65%) during intermediate training as both policies navigate standoff cycles between North and South flank corridors.
        - Attacker advantage emerges as the policy learns to exploit defender anchoring — committing to one flank forces the defender into an unfavorable response.
        - Stable convergence after ~800 episodes.
        """)

    elif selected_page == "🛡️ Dual-Archive K=2":
        st.title("Dual-Archive Co-Evolution — K=2 Flank Perimeter (Phase 6 Equivalent)")
        st.markdown(r"""
        Extension of the Dual-Archive mechanism to the K=2 two-target environment.
        Both agents maintain historical policy snapshots ($\mathcal{H}_A, \mathcal{H}_D$) to prevent policy cycling across North/South flank strategies.
        """)

        st.markdown("---")
        st.subheader("📈 Dual-Archive Training Dynamics & Co-Evolution")
        
        # Display Phase 6 archive curve from docs
        if os.path.exists("results/multi_target_k2/dual_archive_k2_curve.png"):
            display_image("results/multi_target_k2/dual_archive_k2_curve.png")
        elif os.path.exists("results/multi_target_k2/learning_curve_k2.png"):
            display_image("results/multi_target_k2/learning_curve_k2.png")

        st.markdown("---")
        st.subheader("📋 Dual-Archive Methodology & Opponent Regularization")
        st.markdown(r"""
        **Symmetric Opponent Mixture:**
        - **60% Active Self-Play:** Current Attacker vs Current Defender (both policies actively train).
        - **20% Attacker Historical Regularization:** Current Attacker vs Sampled Archived Defender (Attacker learns against past defenses).
        - **20% Defender Historical Regularization:** Sampled Archived Attacker vs Current Defender (Defender learns against past attacks).

        **Strategic Benefits:**
        - Anchors learning against cyclic policy oscillations across chokepoints and flank attractors.
        - Preserves empirical route diversity and suppresses catastrophic forgetting.
        - Drives convergence towards stable Nash equilibrium coverage.
        """)

    elif selected_page == "📊 Robustness (K=2)":
        st.title("K=2 Robustness — Archetype Evaluation Suite")
        st.markdown("""
        Scientific evaluation of K=2 PPO self-play policy against diverse defender archetypes.
        500 paired deterministic episodes per archetype, 95% Wilson Score Confidence Intervals.
        """)

        k2_metrics = load_json_safe_file("results/multi_target_k2/comparison_metrics.json")
        if not k2_metrics or "K_2" not in k2_metrics:
            st.warning("K=2 evaluation metrics not found at `results/multi_target_k2/comparison_metrics.json`.")
        else:
            k2_data = k2_metrics["K_2"]
            k2_learned = k2_data.get("learned_defender", {})
            att_wr = k2_learned.get("attacker_win_rate")
            def_wr = k2_learned.get("defender_win_rate")
            timeout_r = k2_learned.get("timeout_rate")
            avg_steps = k2_learned.get("avg_episode_steps")
            att_ci = k2_learned.get("attacker_ci_95", 0)
            def_ci = k2_learned.get("defender_ci_95", 0)

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Attacker Win vs Learned Def</div>
                    <div class="metric-value">{format_percentage(att_wr)}</div>
                    <div class="metric-subtitle">±{att_ci*100:.1f}% Wilson 95% CI</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Defender Capture Rate</div>
                    <div class="metric-value">{format_percentage(def_wr)}</div>
                    <div class="metric-subtitle">±{def_ci*100:.1f}% Wilson 95% CI</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Timeout Rate</div>
                    <div class="metric-value">{format_percentage(timeout_r)}</div>
                    <div class="metric-subtitle">Unforced Standoffs (T ≥ 75)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Episode Length</div>
                    <div class="metric-value">{format_steps(avg_steps)} steps</div>
                    <div class="metric-subtitle">K=2 Tactical Duration</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 K=2 Robustness — Archetype Breakdown")
        display_image("results/multi_target_k2/k2_robustness_archetype.png")

        # Archetype table
        if k2_metrics and "K_2" in k2_metrics:
            k2_data = k2_metrics["K_2"]
            archetype_map = [
                ("north_camper", "North Camper (Y=1)"),
                ("center_camper", "Center Camper (Y=3)"),
                ("south_camper", "South Camper (Y=5)"),
                ("patroller", "River Patroller"),
            ]
            rows = []
            for arch_key, arch_label in archetype_map:
                m = k2_data.get(arch_key, {})
                rows.append({
                    "Defender Archetype": arch_label,
                    "Attacker Win Rate": format_percentage(m.get("attacker_win_rate")),
                    "Defender Capture": format_percentage(m.get("defender_win_rate")),
                    "Timeout Rate": format_percentage(m.get("timeout_rate")),
                    "Avg Steps": format_steps(m.get("avg_episode_steps")),
                })
            st.subheader("📋 Key Archetype Evaluation Table (K=2)")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("Key 4 tactical archetypes benchmarked for K=2. Values from 500 paired episodes.")

    elif selected_page == "⚖️ vs Single Target":
        st.title("Single-Target Citadel (K=1) vs Multi-Target Flank (K=2)")
        st.markdown("""
        Direct empirical comparison between the **Original Single-Target System (Citadel Baseline, K=1)**
        and the **K=2 Dual-Flank Perimeter System** under 500 paired evaluation episodes.
        """)

        display_image("results/multi_target_k2/k1_vs_k2_comparison.png")

        # Load both metrics files
        single_metrics = load_json_safe_file("results/single_target/comparison_metrics.json")
        k2_metrics = load_json_safe_file("results/multi_target_k2/comparison_metrics.json")
        if not k2_metrics:
            k2_metrics = load_json_safe_file("results/multi_target/comparison_metrics.json")

        if single_metrics and k2_metrics:
            s_learned = single_metrics.get("learned_defender", {})
            k2_data = k2_metrics.get("K_2", {}) if "K_2" in k2_metrics else {}
            m2_learned = k2_data.get("learned_defender", {})

            s_att = s_learned.get("attacker_win_rate")
            m_att = m2_learned.get("attacker_win_rate")
            delta_att = ((m_att - s_att) * 100.0) if (s_att is not None and m_att is not None) else None

            s_def = s_learned.get("defender_win_rate")
            m_def = m2_learned.get("defender_win_rate")
            delta_def = ((m_def - s_def) * 100.0) if (s_def is not None and m_def is not None) else None

            s_steps = s_learned.get("avg_episode_steps")
            m_steps = m2_learned.get("avg_episode_steps")
            delta_steps = (m_steps - s_steps) if (s_steps is not None and m_steps is not None) else None

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-label">Attacker Win Rate Shift</div>
                    <div class="metric-value">{f"{delta_att:+.1f} pp" if delta_att is not None else "N/A"}</div>
                    <div class="metric-subtitle">{format_percentage(s_att)} (K=1) → {format_percentage(m_att)} (K=2)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Defender Capture Collapse</div>
                    <div class="metric-value">{f"{delta_def:+.1f} pp" if delta_def is not None else "N/A"}</div>
                    <div class="metric-subtitle">{format_percentage(s_def)} (K=1) → {format_percentage(m_def)} (K=2)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Episode Duration Shift</div>
                    <div class="metric-value">{f"{delta_steps:+.2f} steps" if delta_steps is not None else "N/A"}</div>
                    <div class="metric-subtitle">{format_steps(s_steps)} steps → {format_steps(m_steps)} steps</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Target Coverage</div>
                    <div class="metric-value">1 → 2</div>
                    <div class="metric-subtitle">Defender Coverage Demands Doubled</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📋 Archetype Comparison Matrix: K=1 vs K=2")
            archetype_list = [
                ("north_camper", "North Camper (Y=1)"),
                ("center_camper", "Center Camper (Y=3)"),
                ("south_camper", "South Camper (Y=5)"),
                ("patroller", "River Patroller"),
            ]
            rows = []
            for arch_key, arch_label in archetype_list:
                s_item = single_metrics.get(arch_key, {})
                m_item = k2_data.get(arch_key, {})
                s_att_r = s_item.get("attacker_win_rate")
                m_att_r = m_item.get("attacker_win_rate")
                d_att = (m_att_r - s_att_r) * 100.0 if (s_att_r is not None and m_att_r is not None) else None
                s_def_r = s_item.get("defender_win_rate")
                m_def_r = m_item.get("defender_win_rate")
                d_def = (m_def_r - s_def_r) * 100.0 if (s_def_r is not None and m_def_r is not None) else None
                rows.append({
                    "Defender Archetype": arch_label,
                    "K=1 Att Win": format_percentage(s_att_r),
                    "K=2 Att Win": format_percentage(m_att_r),
                    "Δ Att Win": f"{d_att:+.1f} pp" if d_att is not None else "N/A",
                    "K=1 Def Capture": format_percentage(s_def_r),
                    "K=2 Def Capture": format_percentage(m_def_r),
                    "Δ Def Capture": f"{d_def:+.1f} pp" if d_def is not None else "N/A",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.caption("Δ = Multi (K=2) minus Single (K=1), in percentage points for the 4 key archetypes.")
        else:
            st.warning("Comparison metrics not available. Run evaluation scripts first.")
            if not single_metrics:
                st.info("Missing: `results/single_target/comparison_metrics.json`")
            if not k2_metrics:
                st.info("Missing: `results/multi_target_k2/comparison_metrics.json`")
