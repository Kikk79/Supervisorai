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
        self.assertGreater(result['score'], 8.0)
        self.assertEqual(result['error_count'], 0)

    def test_bad_code_with_errors(self):
        """Test that code with syntax errors gets a low score and high error count."""
        bad_code = "def my_function(x)\n    print('hello')\n" # Missing colon
        result = self.analyzer.analyze_code(bad_code)

        self.assertIsInstance(result, dict)
        self.assertLess(result['score'], 5.0)
        self.assertGreater(result['error_count'], 0)
        self.assertIn("invalid syntax", result['report'].lower())

    def test_code_with_style_issues(self):
        """Test that code with style issues (e.g., warnings) is scored appropriately."""
        style_issue_code = "def myFunction( my_variable ):\n    \"\"\"Missing module docstring.\"\"\"\n    return my_variable\n"
        result = self.analyzer.analyze_code(style_issue_code)

        self.assertIsInstance(result, dict)
        # Score should be penalized but not as much as a syntax error
        self.assertGreater(result['score'], 5.0)
        self.assertLess(result['score'], 10.0)
        self.assertEqual(result['error_count'], 0) # Style issues are often warnings, not errors
        self.assertGreater(result['warning_count'], 0)
        self.assertIn("invalid-name", result['report'].lower()) # C0103: invalid-name

if __name__ == '__main__':
    unittest.main()
