import unittest
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from reporting.cost_tracker import CostTracker

class TestCostTracker(unittest.TestCase):
    """Test suite for the CostTracker."""

    def setUp(self):
        """Set up a new CostTracker for each test."""
        self.tracker = CostTracker()

    def test_log_call_and_calculate_cost(self):
        """Test that a single call is logged and its cost is calculated correctly."""
        self.tracker.log_call(
            model="claude-3-haiku-20240307",
            input_tokens=1000,
            output_tokens=2000
        )

        self.assertEqual(len(self.tracker.call_history), 1)

        # Expected cost calculation:
        # Input: (1000 / 1,000,000) * $0.25 = $0.00025
        # Output: (2000 / 1,000,000) * $1.25 = $0.0025
        # Total: $0.00275
        expected_cost = 0.00275
        self.assertAlmostEqual(self.tracker.get_total_cost(), expected_cost)
        self.assertAlmostEqual(self.tracker.call_history[0].cost, expected_cost)

    def test_multiple_calls_and_models(self):
        """Test tracking multiple calls with different models."""
        self.tracker.log_call("claude-3-haiku-20240307", 1000, 2000) # Cost: 0.00275
        self.tracker.log_call("claude-3-sonnet-20240229", 5000, 10000) # Cost: (5k/1M)*3 + (10k/1M)*15 = 0.015 + 0.15 = 0.165

        self.assertEqual(len(self.tracker.call_history), 2)
        expected_total_cost = 0.00275 + 0.165
        self.assertAlmostEqual(self.tracker.get_total_cost(), expected_total_cost)

    def test_get_cost_report(self):
        """Test the generation of the cost report."""
        self.tracker.log_call("claude-3-haiku-20240307", 1000, 2000)
        self.tracker.log_call("claude-3-haiku-20240307", 1000, 2000)
        self.tracker.log_call("claude-3-sonnet-20240229", 5000, 10000)

        report = self.tracker.get_cost_report()

        self.assertEqual(report['total_calls'], 3)
        self.assertAlmostEqual(report['total_cost'], (0.00275 * 2) + 0.165)
        self.assertEqual(len(report['cost_by_model']), 2)
        self.assertAlmostEqual(report['cost_by_model']['claude-3-haiku-20240307'], 0.00275 * 2)
        self.assertAlmostEqual(report['cost_by_model']['claude-3-sonnet-20240229'], 0.165)

    def test_unknown_model_uses_default_pricing(self):
        """Test that an unknown model falls back to default pricing."""
        self.tracker.log_call("unknown-model-xyz", 10000, 10000)

        # Default pricing: Input $1/M, Output $5/M
        # (10k/1M)*1 + (10k/1M)*5 = 0.01 + 0.05 = 0.06
        expected_cost = 0.06
        self.assertAlmostEqual(self.tracker.get_total_cost(), expected_cost)

if __name__ == '__main__':
    unittest.main()
