import streamlit as st
import json
import os
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Red Queen Tactical Arena", initial_sidebar_state="collapsed")

# Custom CSS: Permanently remove Streamlit header & sidebar, make horizontal navbar sticky
st.markdown("""
<style>
    /* 1. Permanently hide Streamlit top header & sidebar */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebar"], 
    [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    /* 2. Optimize main container for 100% zoom screens */
    .block-container {
        padding-top: 0.35rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        max-width: 100% !important;
    }
    
    /* 3. STICKY HORIZONTAL NAVBAR AT THE VERY TOP (ZERO OVERLAP!) */
    div[data-testid="stRadio"] {
        position: sticky !important;
        top: 0px !important;
        z-index: 99999 !important;
        background: rgba(15, 23, 42, 0.96) !important;
        backdrop-filter: blur(16px) !important;
        padding: 5px 10px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5) !important;
        margin-bottom: 6px !important;
    }
    
    /* 4. Single-line horizontal flex layout (never wraps off-screen!) */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
    }
    
    /* 5. Nav Pills Styling */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 5px 11px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        font-size: 0.8rem !important;
        white-space: nowrap !important;
    }
    
    /* Hide the default radio circle indicator */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        border-color: rgba(0, 210, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Active Tab Highlight (Neon Cyan Gradient) */
    div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, rgba(0, 210, 255, 0.25), rgba(0, 119, 255, 0.25)) !important;
        border-color: #00d2ff !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 0 12px rgba(0, 210, 255, 0.4) !important;
    }
    
    /* 6. Optimal graph sizing (prevents stretched/zoomed-in graphs on wide screens) */
    div[data-testid="stImage"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        margin-top: 4px !important;
        margin-bottom: 12px !important;
    }
    div[data-testid="stImage"] > img {
        max-width: 780px !important;
        max-height: 460px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

def display_image(image_path):
    if os.path.exists(image_path):
        col_l, col_m, col_r = st.columns([1.5, 7, 1.5])
        with col_m:
            st.image(image_path, use_container_width=True)
    else:
        st.warning(f"Image not found: {image_path}. Please run the training script first.")

# Sticky Top Horizontal Navbar with concise names fitting single row on 100% zoom
PAGES = [
    "🎮 3D Tactical Arena",
    "📜 Game Theory", 
    "🎯 Fixed Baseline", 
    "🔄 Red Queen", 
    "🛡️ Double Archive", 
    "📊 Robustness",
    "🌍 Real-World"
]

page = st.radio(
    "Navigation Bar",
    PAGES,
    horizontal=True,
    label_visibility="collapsed"
)

def get_all_replays_data():
    """Load each replay JSON file, parse it, and re-serialize with json.dumps.
    Re-serializing ensures special characters are properly escaped before the
    data is embedded inside a <script> block (fix for issue #12)."""
    def load_json_safe(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.dumps(json.load(f))  # parse then re-encode safely
                except json.JSONDecodeError:
                    return "null"
        return "null"
    return (
        load_json_safe("replays/fixed_opponent.json"),
        load_json_safe("replays/red_queen.json"),
        load_json_safe("replays/red_queen_history.json")
    )

if page == "🎮 3D Tactical Arena":
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
        components.html(injected, height=680, scrolling=False)
    else:
        st.warning("3D dashboard file not found.")

elif page == "📜 Game Theory":
    if os.path.exists("docs/phase1_game_theory_spec.md"):
        with open("docs/phase1_game_theory_spec.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Theory document not found.")

elif page == "🎯 Fixed Baseline":
    st.title("Fixed Opponent Baseline (PPO)")
    st.markdown("""
    In this setup, the Defender's policy is fixed (stationary baseline) to benchmark single-agent PPO navigation across the 11x7 'Three Bridges' map.
    
    Equipped with a distance gradient and a proximity **Danger Bubble** to steer around the Defender's position, the Attacker rapidly converges to an optimal, decisive pathfinding strategy.
    """)
    display_image("docs/phase3_training_curve.png")

elif page == "🔄 Red Queen":
    st.title("Full Multi-Agent RL: The Red Queen Effect")
    st.markdown(r"""
    Both agents train simultaneously using PPO. Forced to route through the 3 bridges, their policies change in response to each other over training.

    **Hypothesis:** The oscillating win-rate curves are consistent with a *Red Queen dynamic* — a cycle where each agent adapts to exploit the other, only to be countered in turn. However, a smoothed win-rate curve alone cannot rule out simpler explanations such as stochastic policy noise or coincident learning-rate schedules.
    
    **Strictly Zero-Sum Formulation:** The environment and learning dynamics are strictly zero-sum ($R_{\text{Attacker}} + R_{\text{Defender}} \equiv 0$ at every transition). Distance shaping $F(s, s')$ is applied symmetrically: $+F$ to the Attacker and $-F$ to the Defender. Under potential-based reward shaping (Ng et al. 1999; Devlin & Kudenko 2011), this guarantees zero-sum payoff invariance and preserves Nash equilibrium structures.
    
    A confirmed Red Queen cycle would require tracking *which* bridge is chosen over time and showing that the Defender's strategy shifts in response — not just that win rates fluctuate.
    """)
    display_image("docs/phase4_marl_curve.png")

elif page == "🛡️ Double Archive":
    st.title("Double History Archive (Stability Hypothesis)")
    st.markdown("""
    Both agents train against a rolling archive of past opponent policies (60% current vs current, 20% each archive matchup).

    **Hypothesis:** Training against historical opponents should reduce the severity of catastrophic forgetting and push both policies toward a more *generalised* mixed strategy, ideally converging toward a Nash Equilibrium corridor.

    **What the chart shows:**
    - **Top panel:** Rolling attacker *and* defender win rates — plotted symmetrically so you can see both sides of the competition. A flatter, more stable curve compared to Phase 4 is consistent with reduced forgetting.
    - **Bottom panel:** Timeout rate (fraction of episodes that end without either agent winning). High timeouts mean the attacker is failing to find a path, *not* that the defender is actively catching it — an important distinction.

    Establishing a Nash Equilibrium claim would require measuring exploitability against a full population of strategies — a single training curve is not sufficient.

    Methodology follows the spirit of opponent-archive approaches studied in the MARL literature (e.g., Policy-Space Response Oracles, Lanctot et al. 2017).
    """)
    display_image("docs/phase6_archive_curve.png")

    st.markdown("---")
    st.subheader("🔬 Empirical Evidence: Bridge-Route Diversity & Catastrophic Forgetting")
    st.markdown(r"""
    To rigorously demonstrate that the Double History Archive prevents catastrophic forgetting, we measured crossing distributions across all 3 bridges over 150 deterministic test episodes:
    - **Theoretical Maximum Shannon Entropy:** $1.585$ bits (perfect 33.3% equal exploration).
    - **Naive Self-Play (Phase 4):** Entropy collapses to **1.470 bits (92.8%)** with heavy North bias (46.0%) and South neglect (16.1%).
    - **Double History Archive (Phase 6):** Entropy reaches **1.567 bits (98.9% of theoretical maximum)** with uniform exploration (26.7% North, 34.0% Center, 39.3% South) and **0.0% timeouts**.
    """)
    display_image("docs/bridge_diversity_chart.png")

elif page == "📊 Robustness":
    st.title("Robustness & Empirical Game Theory")
    st.markdown("""
    The final Attackers from **Red Queen Self-Play (Phase 4)** and **Double History Archive (Phase 6)** are evaluated against **4 hand-coded, rule-based defender archetypes** (North Camper, Center Camper, South Camper, River Patroller).

    Evaluation uses **deterministic greedy actions** and **paired random seeds** so both agents face identical starting positions. 95% confidence intervals are reported.
    """)
    display_image("docs/robustness_chart.png")

    st.markdown("---")
    st.subheader("📐 Empirical Game-Theoretic Analysis (EGTA): Minimax Exploitability")
    st.markdown(r"""
    Win rate alone does not prove convergence to a Nash equilibrium — an agent could achieve high win rates against average opponents while remaining fatally vulnerable to counter-strategies.
    
    Using Empirical Game-Theoretic Analysis (EGTA), we computed the security level $v_{\text{sec}}$ (worst-case performance against optimal counter-strategies) and exploitability $\delta = v^* - v_{\text{sec}}$ across the spatial dilemma meta-game suite:
    - **Phase 4 (Naive Self-Play):** Collapses to **7.3% win rate** against River Patroller (Exploitability $\delta = 0.160$).
    - **Phase 6 (Double Archive):** Robust floor of **31.3% worst-case win rate** (Exploitability $\delta = 0.140$) and +0.227 average margin across all opponents.
    """)
    display_image("docs/nash_exploitability_chart.png")

elif page == "🌍 Real-World":
    st.title("🌍 Domain-Agnostic Game Theory & Real-World Impact")
    st.markdown("""
    The **Three Bridges Dilemma** is a computational model for zero-sum competition
    where an adaptive infiltrator must bypass an adaptive defender through bottleneck channels.

    In real-world competitive systems, naive adaptive learning can cause **policy cycling** —
    overreacting to recent threats while leaving previously exploited vectors re-exposed.
    Opponent-archive methods are motivated by the hypothesis that training against a diverse history
    of past policies produces more *generalised*, harder-to-exploit strategies.

    The domain analogies below illustrate where these dynamics may apply. They are *conceptual mappings*,
    not direct empirical claims from this project.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💻 1. Cybersecurity: AI Firewall vs Autonomous Malware")
        st.markdown("""
        * **The Attacker (🔴):** Self-mutating Malware, zero-day bot, or advanced persistent threat (APT).
        * **The Defender (🔵):** Adaptive AI Firewall / Intrusion Detection System (IDS).
        * **The Bottlenecks (3 Bridges):** Primary attack surfaces (Port 80 HTTP, Port 443 HTTPS, Port 22 SSH).
        * **The Citadel (🎯):** Core corporate database containing sensitive credentials.
        
        **The Naive Trap (Red Queen Cycling):**  
        A hacker breaches Port 80. The AI firewall updates its weights to aggressively monitor Port 80. 
        The hacker pivots to Port 443. The firewall hyper-focuses on 443 and *forgets* to guard Port 80. 
        Next week, the hacker simply re-deploys the old Port 80 exploit, and the firewall collapses.
        
        **The Double Archive Solution (Nash Equilibrium):**  
        The AI firewall is trained against an archive of *all historical attack signatures simultaneously*. 
        It learns generalized anomaly detection across all ports at once, making it impossible for malware to exploit cyclical blind spots.
        """)

    with col2:
        st.subheader("🧬 2. Medicine & Oncology: Cancer vs Combination Therapy")
        st.markdown("""
        * **The Attacker (🔴):** Mutating Superbug, bacterial pathogen, or malignant tumor cells.
        * **The Defender (🔵):** Targeted pharmaceutical drug or engineered medical nanobots.
        * **The Bottlenecks (3 Bridges):** Metabolic pathways, cell wall receptors, or blood vessels.
        * **The Citadel (🎯):** Host organ tissue / vital DNA replication machinery.
        
        **The Biological Red Queen Effect:**  
        An oncologist treats cancer with Drug A. The tumor mutates and develops resistance to Drug A. 
        The doctor switches to Drug B. The tumor adapts to Drug B, but as an evolutionary metabolic tradeoff, loses resistance to Drug A. 
        Sequential mono-therapy lets the disease run indefinitely.
        
        **The Double Archive Solution (Nash Equilibrium):**  
        Maps directly to **Combination & Adaptive Therapy**. By applying historical combinations or rotating archived drug regimens, 
        doctors trap the mutating pathogen in an evolutionary dead-end where it cannot mutate fast enough to survive all vectors simultaneously.
        """)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🚗 3. Autonomous Driving: Defensive Vehicle Control")
        st.markdown("""
        * **The Attacker (🔴):** Ego vehicle navigating safely to its exit or drop-off zone.
        * **The Defender (🔵):** Adversarial traffic, erratic drivers, and unexpected bottlenecks.
        * **The Bottlenecks (3 Bridges):** 3 highway lanes / bridge lanes during heavy congestion.
        * **The Citadel (🎯):** Destination off-ramp / safe highway exit.
        
        **The Naive Trap:**  
        If an autonomous vehicle only trains against the driving habits of cars in its immediate recent memory, 
        it overfits to polite commuters. When it encounters an aggressive driver cutting it off, the policy freezes or causes collisions.
        
        **The Double Archive Solution (Nash Equilibrium):**  
        Evaluating against an archive of historical driver archetypes (aggressive tailgaters, erratic lane changers, timid mergers) 
        forces the self-driving agent to learn an unexploitable defensive policy that remains calm and safe regardless of who shares the road.
        """)

    with col4:
        st.subheader("📈 4. Algorithmic Finance: Market Makers vs Arbitrageurs")
        st.markdown("""
        * **The Attacker (🔴):** High-frequency statistical arbitrage bots seeking liquidity imbalances.
        * **The Defender (🔵):** Automated Market Maker (AMM) providing bid-ask liquidity.
        * **The Bottlenecks (3 Bridges):** Liquidity pools, currency routing pairs, or exchange order books.
        * **The Citadel (🎯):** Maximum alpha extraction / total risk-free arbitrage.
        
        **The Naive Trap:**  
        Trading algorithms that optimize only for recent short-term volatility regimes get caught in catastrophic flash crashes 
        when market conditions abruptly flip from bull to bear.
        
        **The Double Archive Solution (Nash Equilibrium):**  
        Game-theoretic multi-agent training pits market makers against historical archives of liquidity shocks and flash crashes, 
        discovering equilibrium pricing strategies that are resilient across macroeconomic regime shifts.
        """)

    st.markdown("---")

    st.subheader("📊 Strategic Comparison Matrix")
    st.markdown("""
    | Domain | Infiltrator Vector | Defensive Shield | Bottleneck Channels | Potential Advantage (Hypothesis) |
    | :--- | :--- | :--- | :--- | :--- |
    | **Cybersecurity** | Malware / APT | Adaptive AI Firewall | Ports 80, 443, 22 | More robust to rehashed historical exploits |
    | **Oncology & Medicine** | Mutating Tumor | Combination Therapy | Receptors / Metabolic Pathways | Combination therapy traps pathogen in evolutionary dead-ends |
    | **Autonomous Driving** | Ego Vehicle Navigation | Adversarial Road Traffic | 3 Highway Lanes | Exposure to diverse driver archetypes may improve generalization |
    | **Quantitative Finance**| Arbitrageur Bot | Automated Market Maker | Liquidity Pools / Order Books | Historical regime training may improve resilience to flash crashes |
    """)

    st.info("""
    📌 **Research context:** Naive competitive self-play can produce cycling rather than convergence.
    Opponent-archive methods (e.g., Policy-Space Response Oracles, Fictitious Play) are studied precisely
    because they are motivated to push agents toward more general strategies.
    Establishing a Nash Equilibrium claim requires exploitability measurement across a defined policy population —
    single training curves are not sufficient evidence.
    """)
