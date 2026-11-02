import unittest
import os
import sys
from streamlit.testing.v1 import AppTest

DASHBOARD_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "dashboard.py"))

class TestDashboardModes(unittest.TestCase):
    """Test suite verifying Streamlit dashboard mode-switching and K=2 multi-target interface."""

    def test_dashboard_initial_run(self):
        """Test that dashboard loads initially in Single-Target mode without exception."""
        at = AppTest.from_file(DASHBOARD_FILE, default_timeout=30)
        at.run()
        self.assertEqual(len(at.exception), 0, f"Dashboard initial load raised exception: {at.exception}")
        self.assertEqual(at.session_state["dashboard_mode"], "single")

    def test_single_target_navigation_options(self):
        """Verify navigation segmented control has all 7 original single-target pages."""
        at = AppTest.from_file(DASHBOARD_FILE, default_timeout=30)
        at.run()
        
        self.assertGreater(len(at.button_group), 0)
        nav = at.button_group[0]
        self.assertEqual(nav.value, "🎮 3D Arena")

    def test_single_target_pages_functional(self):
        """Verify all original single target pages still load cleanly without errors."""
        at = AppTest.from_file(DASHBOARD_FILE, default_timeout=30)
        at.run()
        nav = at.button_group[0]
        
        pages_to_test = [
            "📜 Game Theory", 
            "🎯 Baseline PPO", 
            "🔄 Naive Self-Play", 
            "🛡️ Dual-Archive", 
            "📊 Robustness (EGTA)", 
            "🌍 Applications"
        ]
        
        for page in pages_to_test:
            nav.set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"Single Target Page {page} failed: {at.exception}")

    def test_switch_to_multi_target_k2_mode(self):
        """Test switching mode to Multi-Target (K=2) and verify separate sub-navigation and pages."""
        at = AppTest.from_file(DASHBOARD_FILE, default_timeout=30)
        at.run()

        # Find and click the Multi-Target mode button
        btn_multi = None
        for btn in at.button:
            if "Multi-Target" in btn.label:
                btn_multi = btn
                break
        self.assertIsNotNone(btn_multi, "Multi-Target switcher button not found")
        btn_multi.click().run()

        self.assertEqual(len(at.exception), 0, f"Switching mode raised exception: {at.exception}")
        self.assertEqual(at.session_state["dashboard_mode"], "multi_k2")

        # In K=2 mode, default page should be 🎮 3D Arena (K=2)
        nav = at.button_group[0]
        self.assertEqual(nav.value, "🎮 3D Arena (K=2)")

        # Test sub-pages of K=2
        k2_pages = [
            "🎯 Fixed Baseline",
            "🔄 Red Queen Self-Play",
            "🛡️ Dual-Archive K=2",
            "📊 Robustness (K=2)",
            "⚖️ vs Single Target"
        ]
        for page in k2_pages:
            nav.set_value(page).run()
            self.assertEqual(len(at.exception), 0, f"K=2 Page {page} failed: {at.exception}")

    def test_k2_files_and_assets_exist(self):
        """Verify dedicated folders and files for K=2 exist."""
        self.assertTrue(os.path.exists("dashboard_3d_k2.html"), "dashboard_3d_k2.html is missing")
        self.assertTrue(os.path.exists("replays/multi_target_k2/k2_ppo_selfplay.json"), "K=2 replay file missing")
        has_att = os.path.exists("models/multi_target_k2/ppo_k2_attacker.pth") or os.path.exists("models/multi_target/ppo_k2_attacker.pth")
        has_def = os.path.exists("models/multi_target_k2/ppo_k2_defender.pth") or os.path.exists("models/multi_target/ppo_k2_defender.pth")
        self.assertTrue(has_att, "K=2 attacker model missing")
        self.assertTrue(has_def, "K=2 defender model missing")
        self.assertTrue(os.path.exists("results/multi_target_k2/learning_curve_k2.png"), "K=2 learning curve missing")
        self.assertTrue(os.path.exists("results/multi_target_k2/k2_robustness_archetype.png"), "K=2 robustness chart missing")
        self.assertTrue(os.path.exists("results/multi_target_k2/k1_vs_k2_comparison.png"), "K=1 vs K=2 chart missing")

if __name__ == "__main__":
    unittest.main()
