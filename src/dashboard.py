import streamlit as st
import json
import os
import streamlit.components.v1 as components

st.set_page_config(
    layout="wide",
    page_title="Red Queen MARL | SVNIT Surat",
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
        padding: 7px 12px !important;
        font-size: 0.81rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
        white-space: nowrap !important;
        transition: all 0.16s ease-in-out !important;
        cursor: pointer !important;
        text-align: center !important;
        outline: none !important;
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
    <div class="masthead-right">
        <div class="masthead-affiliation">SVNIT Surat • Department of Mathematics</div>
        <div class="masthead-authors">Authors: <strong>Navadeep G.</strong>, <strong>Brij R.</strong>, <strong>Shrey M.</strong>, <strong>Shree P.</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

PAGES = [
    "🎮 3D Arena",
    "📜 Game Theory", 
    "🎯 Baseline PPO", 
    "🔄 Naive Self-Play", 
    "🛡️ Dual-Archive", 
    "📊 Robustness (EGTA)",
    "🌍 Applications"
]

# Modern Native Streamlit Segmented Control (Zero Radio Dots, Zero Text Wrapping)
selected_page = st.segmented_control(
    "Navigation",
    PAGES,
    default=PAGES[0],
    selection_mode="single",
    required=True,
    label_visibility="collapsed",
    width="stretch"
)

if not selected_page:
    selected_page = PAGES[0]

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

# -------------------------------------------------------------
# PAGE ROUTING
# -------------------------------------------------------------

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

    # Metric Cards
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

    # Metric Cards
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
