import tempfile
import os
from io import StringIO
from pylint.lint import Run
from pylint.reporters.text import TextReporter

class CodeQualityAnalyzer:
    """
    Analyzes a string of Python code using pylint and provides a quality score.
    """

    def analyze_code(self, code_string: str) -> dict:
        """
        Lints the given code string and returns a dictionary with the analysis.

        Args:
            code_string: A string containing the Python code to analyze.

        Returns:
            A dictionary containing 'score', 'error_count', and 'report'.
        """
        if not code_string.strip():
            return {"score": 10.0, "error_count": 0, "report": "No code provided."}

        # pylint requires a file to lint, so we create a temporary one.
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code_string)
            temp_filepath = temp_file.name

        # Capture pylint's output
        pylint_output = StringIO()
        reporter = TextReporter(pylint_output)

        try:
            # Run pylint on the temporary file
            results = Run([temp_filepath], reporter=reporter, exit=False)

            # The reporter captures the text report.
            report_text = pylint_output.getvalue()

            # The results object has stats.
            stats = results.linter.stats
            error_count = getattr(stats, 'error', 0)
            warning_count = getattr(stats, 'warning', 0)

            # Pylint's score is out of 10.
            # We can use the score directly, or calculate our own.
            # Let's use pylint's score. It's usually present in the report text.
            score = self._parse_score_from_report(report_text)

            return {
                "score": score,
                "error_count": error_count,
                "warning_count": warning_count,
                "report": report_text
            }
        finally:
            # Clean up the temporary file
            os.unlink(temp_filepath)

    def _parse_score_from_report(self, report_text: str) -> float:
        """
        Parses the final score from the pylint text report.
        Example line: "Your code has been rated at 10.00/10"
        """
        import re
        match = re.search(r"Your code has been rated at (-?\d+\.\d+)/10", report_text)
        if match:
            return float(match.group(1))
        # If no score is found (e.g., due to a fatal error), return a low score.
        return 0.0
