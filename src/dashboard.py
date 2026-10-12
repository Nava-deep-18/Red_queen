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
        margin-top: 10px !important;
        margin-bottom: 25px !important;
    }
    div[data-testid="stImage"] > img {
        max-width: 860px !important;
        width: auto !important;
        height: auto !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)

def display_image(image_path):
    if os.path.exists(image_path):
        col_l, col_m, col_r = st.columns([1, 8, 1])
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
    def load_json(path):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        return "{}"
    return (
        load_json("replays/fixed_opponent.json"),
        load_json("replays/red_queen.json"),
        load_json("replays/red_queen_history.json")
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
    st.title("Game Theory & The Red Queen Hypothesis")
    if os.path.exists("docs/phase1_game_theory_spec.md"):
        with open("docs/phase1_game_theory_spec.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())
    else:
        st.warning("Theory document not found.")

elif page == "🎯 Fixed Baseline":
    st.title("Fixed Opponent Baseline (PPO)")
    st.markdown("In this setup, we paralyze the Defender and train the Attacker using PPO to navigate the 11x7 'Three Bridges' map. A distance-based scent trail allows it to quickly achieve a high win rate.")
    display_image("docs/phase3_training_curve.png")

elif page == "🔄 Red Queen":
    st.title("Full Multi-Agent RL: The Red Queen Effect")
    st.markdown("Both agents train simultaneously using PPO. Forced through the 3 bridges, they enter a Rock-Paper-Scissors cycle. Notice the massive jagged sine-waves as they endlessly chase each other and suffer from **Catastrophic Forgetting**.")
    display_image("docs/phase4_marl_curve.png")

elif page == "🛡️ Double Archive":
    st.title("Double History Archive (Nash Equilibrium)")
    st.markdown("Agents are forced to fight historical ghosts from their Archive. This cures Catastrophic Forgetting and forces them into a generalized, robust mixed strategy, stabilizing both agents into a calm, persistent Nash Equilibrium corridor.")
    display_image("docs/phase6_archive_curve.png")
    
elif page == "📊 Robustness":
    st.title("Robustness Comparison: The True Test")
    st.markdown("We take the final Attackers from **Red Queen Self-Play** and **Double History Archive** and evaluate them against **4 Hardcoded Archetypes** (North Camper, Center Camper, South Camper, and River Patroller).")
    display_image("docs/robustness_chart.png")
    st.info("💡 **Results:** The naive self-play attacker overfits to whatever bridge it used last in training, causing it to fail against other campers. The Double Archive attacker retains its memory of all 3 bridges and smoothly navigates around any archetype!")

elif page == "🌍 Real-World":
    st.title("🌍 Domain-Agnostic Game Theory & Real-World Impact")
    st.markdown("""
    The **Three Bridges Dilemma** is not just an abstract grid game. It is a mathematical model for any zero-sum competition 
    where an adaptive infiltrator must bypass an adaptive defender through bottleneck channels.
    
    In real-world competitive systems, naive adaptive learning causes **Catastrophic Forgetting**—overreacting to yesterday's threat while leaving 
    past vectors completely exposed. **Double Opponent Archive** solves this by converging to an unexploitable **Nash Equilibrium**.
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
    | Domain | Infiltrator Vector | Defensive Shield | Bottleneck Channels | The Nash Equilibrium Advantage |
    | :--- | :--- | :--- | :--- | :--- |
    | **Cybersecurity** | Malware / APT | Adaptive AI Firewall | Ports 80, 443, 22 | Immune to rehashed historical exploits |
    | **Oncology & Medicine** | Mutating Tumor | Combination Therapy | Receptors / Metabolic Pathways | Traps pathogen in an evolutionary dead-end |
    | **Autonomous Driving** | Ego Vehicle Navigation | Adversarial Road Traffic | 3 Highway Lanes | Universally robust defensive driving |
    | **Quantitative Finance**| Arbitrageur Bot | Automated Market Maker | Liquidity Pools / Order Books | Resilient against sudden market flash crashes |
    """)

    st.success("🎯 **Key Takeaway:** Naive competitive learning produces an overreactive amateur who forgets yesterday's lessons. Double History Archive produces an **unexploitable grandmaster** with permanent general robustness.")
