import unittest
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from analysis.code_analyzer import CodeQualityAnalyzer

class TestCodeQualityAnalyzer(unittest.TestCase):
    """Test suite for the CodeQualityAnalyzer."""

    def setUp(self):
        """Set up the analyzer."""
        self.analyzer = CodeQualityAnalyzer()

    def test_good_code_analysis(self):
        """Test that well-formatted code receives a high score."""
        good_code = "def my_function(x):\n    \"\"\"This is a docstring.\"\"\"\n    return x + 1\n"
        result = self.analyzer.analyze_code(good_code)

        self.assertIsInstance(result, dict)
        # Pylint is strict, even this 'good' code has minor convention warnings (e.g. module docstring)
        self.assertGreater(result['score'], 4.0)
        self.assertEqual(result['error_count'], 0)

    def test_bad_code_with_errors(self):
        """Test that code with syntax errors gets a low score and high error count."""
        bad_code = "def my_function(x)\n    print('hello')\n" # Missing colon
        result = self.analyzer.analyze_code(bad_code)

        self.assertIsInstance(result, dict)
        self.assertEqual(result['score'], 0.0)
        self.assertEqual(result['error_count'], 1)
        self.assertIn("expected ':'", result['report'].lower())

    def test_code_with_style_issues(self):
        """Test that code with style issues (e.g., warnings) is scored appropriately."""
        # This code has an unused variable, which should trigger a warning.
        style_issue_code = "def my_func():\n    \"\"\"A function with a warning.\"\"\"\n    x = 10 # Unused variable\n    return 5\n"
        result = self.analyzer.analyze_code(style_issue_code)

        self.assertIsInstance(result, dict)
        self.assertLess(result['score'], 10.0) # Score should be less than perfect
        self.assertEqual(result['error_count'], 0) # Style issues are often warnings, not errors
        self.assertGreater(result['warning_count'], 0)
        self.assertIn("unused-variable", result['report'].lower()) # W0612: unused-variable

if __name__ == '__main__':
    unittest.main()
