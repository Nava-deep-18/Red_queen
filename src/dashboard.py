import streamlit as st
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import time

st.set_page_config(page_title="Red Queen AI Arena", layout="wide")

st.markdown("""
    <style>
           .block-container {
                padding-top: 2rem;
                padding-bottom: 0rem;
            }
    </style>
    """, unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["🎥 Visual Replays", "Fixed Opponent", "Red Queen", "Red Queen + History"])

def display_image(path):
    if os.path.exists(path):
        img = Image.open(path)
        st.image(img, use_container_width=True)
    else:
        st.warning(f"Training results not yet generated for {path}. Run the respective training script first.")

# Show big headers only on chart pages to save space for the game board
if page != "🎥 Visual Replays":
    st.title("🔴 Red Queen AI Arena 🔵")
    st.markdown("""
    A Game-Theoretic Multi-Agent Reinforcement Learning Project studying Co-Evolution, 
    Adaptation, and the Red Queen Effect in Attacker vs. Defender Agents.
    """)
    st.divider()

# Page Logic
if page == "Fixed Opponent":
    st.header("Fixed Opponent (DQN Baseline)")
    st.markdown("Attacker trains with DQN while the Defender remains fixed (stands still).")
    display_image("docs/phase3_training_curve.png")

elif page == "Red Queen":
    st.header("The Red Queen Effect (Full MARL)")
    st.markdown("Both Attacker and Defender train simultaneously with PPO. Watch for oscillating win-rates (The Red Queen Effect).")
    display_image("docs/phase4_marl_curve.png")

elif page == "Red Queen + History":
    st.header("Red Queen + History (Opponent Archive)")
    st.markdown("Both learn, but are evaluated/trained against an archive of past opponent versions to improve robustness.")
    display_image("docs/phase6_archive_curve.png")

elif page == "🎥 Visual Replays":
    st.title("🎥 Visual Replay Viewer")
    st.markdown("Watch the agents' strategies evolve over training epochs!")
    st.markdown("<br>", unsafe_allow_html=True)
    
    experiment_map = {
        "Fixed Opponent": "replays/fixed_opponent.json",
        "Red Queen": "replays/red_queen.json",
        "Red Queen + History": "replays/red_queen_history.json"
    }
    
    # Split screen into Game Board (Left) and Controls (Right)
    col_board, col_controls = st.columns([2.5, 1])
    
    with col_controls:
        st.markdown("### 🎮 Replay Controls")
        selected_exp = st.selectbox("Select Experiment", list(experiment_map.keys()))
        replay_file = experiment_map[selected_exp]
        
        if os.path.exists(replay_file):
            with open(replay_file, 'r') as f:
                replay_data = json.load(f)
                
            episodes_list = list(replay_data["episodes"].keys())
            max_ep_num = len(episodes_list)
            selected_ep_num = st.number_input(f"Select Episode (1 to {max_ep_num})", min_value=1, max_value=max_ep_num, value=max_ep_num)
            selected_ep = str(selected_ep_num)
            trajectory = replay_data["episodes"][selected_ep]
            max_steps = len(trajectory)
            
            step_idx = st.slider("Step", 0, max_steps - 1, 0, key="slider")
            st.markdown("<br>", unsafe_allow_html=True)
            play_btn = st.button("▶ Play Animation", use_container_width=True)
        else:
            st.warning("No data found.")
            play_btn = False

    with col_board:
        if os.path.exists(replay_file):
            board_placeholder = st.empty()
            
            def render_board(step_index):
                grid_w, grid_h = replay_data["grid_size"]
                obstacles = replay_data["obstacles"]
                target = replay_data["target"]
                
                curr_state = trajectory[step_index]
                ax, ay = curr_state["attacker"]
                dx, dy = curr_state["defender"]
                
                grid = [['·' for _ in range(grid_w)] for _ in range(grid_h)]
                for obs in obstacles:
                    grid[obs[1]][obs[0]] = '🧱'
                grid[target[1]][target[0]] = '🎯'
                
                status_text = f"Status: Game Ongoing... (Step {step_index})"
                status_color = "#0d6efd" # Blue
                
                if ax == dx and ay == dy:
                    grid[dy][dx] = '💥'
                    status_text = "Result: Defender caught the attacker! 🔵🛡️"
                    status_color = "#dc3545" # Red
                else:
                    grid[dy][dx] = '🔵'
                    grid[ay][ax] = '🔴'
                    if ax == target[0] and ay == target[1]:
                        status_text = "Result: Attacker reached the target! 🔴🏆"
                        status_color = "#198754" # Green
                    
                # The board HTML with margin: 0 auto to perfectly center it within this column
                grid_html = "<div style='font-family: monospace; font-size: 38px; line-height: 1.2; text-align: center; letter-spacing: 15px; background: #1e1e1e; padding: 25px; border-radius: 10px; width: fit-content; margin: 0 auto;'>"
                for row in grid:
                    grid_html += "".join(row) + "<br>"
                    
                grid_html += f"<div style='font-family: sans-serif; font-size: 16px; letter-spacing: normal; margin-top: 15px; padding: 10px; border-radius: 5px; background: {status_color}; color: white;'>{status_text}</div>"
                grid_html += "</div>"
                
                board_placeholder.markdown(grid_html, unsafe_allow_html=True)
            
            if play_btn:
                for i in range(max_steps):
                    render_board(i)
                    time.sleep(0.3)
            else:
                render_board(step_idx)
