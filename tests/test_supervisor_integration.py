import unittest
from unittest.mock import MagicMock, AsyncMock
import asyncio
import sys
import os
import tempfile
import shutil

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from supervisor_agent.core import SupervisorCore
from supervisor_agent import InterventionLevel

class TestSupervisorCodeIntegration(unittest.TestCase):
    """
    Integration test for the code-aware supervisor feature.
    """

    def setUp(self):
        """Set up a SupervisorCore instance for testing."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.temp_dir = tempfile.mkdtemp(prefix="supervisor_code_test_")

        # Mock the LLMManager
        mock_llm_client = MagicMock()
        # Configure the mock query to return a valid response structure
        mock_llm_client.query = AsyncMock(return_value={
            "content": {"overall_score": 0.9, "reasoning": "Mocked.", "is_safe": True},
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
        mock_llm_manager = MagicMock()
        mock_llm_manager.get_client.return_value = mock_llm_client

        self.supervisor = SupervisorCore(data_dir=self.temp_dir, llm_manager=mock_llm_manager)
        self.loop.run_until_complete(self.supervisor._load_knowledge_base())

    def tearDown(self):
        """Clean up after tests."""
        self.loop.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_supervisor_intervenes_on_bad_code(self):
        """
        Test that the supervisor intervenes when presented with low-quality Python code.
        """
        async def run_test():
            task_id = await self.supervisor.monitor_agent(
                agent_name="coding_agent",
                framework="test",
                task_input="write a function",
                instructions=["write a function that returns 1"]
            )

            # This code has a syntax error and multiple style issues.
            bad_code = "def my_func( x):\n  y=x+1\n  return y"

            result = await self.supervisor.validate_output(
                task_id,
                bad_code,
                output_type="python_code"
            )

            # The intervention should be required due to the low code quality score
            intervention = result['intervention_result']
            self.assertTrue(intervention['intervention_required'])
            # Expect a correction or escalation due to high error count
            self.assertIn(intervention['level'], [InterventionLevel.CORRECTION.value, InterventionLevel.ESCALATION.value])

        self.loop.run_until_complete(run_test())

    def test_supervisor_allows_good_code(self):
        """
        Test that the supervisor allows high-quality Python code.
        """
        async def run_test():
            task_id = await self.supervisor.monitor_agent(
                agent_name="coding_agent",
                framework="test",
                task_input="write a function",
                instructions=["write a function that returns 1"]
            )

            good_code = "def my_func(x):\n    \"\"\"A simple function.\"\"\"\n    y = x + 1\n    return y\n"

            result = await self.supervisor.validate_output(
                task_id,
                good_code,
                output_type="python_code"
            )

            # No intervention should be required for high-quality code
            intervention = result['intervention_result']
            self.assertFalse(intervention['intervention_required'])

        self.loop.run_until_complete(run_test())

    @patch('supervisor_agent.llm_judge.LLMJudge.evaluate_output', new_callable=AsyncMock)
    def test_supervisor_handles_image_output(self, mock_evaluate_output):
        """
        Test that the supervisor correctly passes an image URL to the LLM Judge.
        """
        # --- Setup ---
        # Configure the mock judge to return a successful evaluation
        mock_evaluate_output.return_value = {
            "overall_score": 0.9, "reasoning": "Looks good.", "is_safe": True
        }

        async def run_test():
            image_url = "http://example.com/image.png"
            task_id = await self.supervisor.monitor_agent("img_agent", "test", "gen_image", [])

            # --- Execute ---
            await self.supervisor.validate_output(
                task_id,
                output=image_url,
                output_type="image"
            )

            # --- Assertions ---
            # Check that the judge was called with the correct parameters
            mock_evaluate_output.assert_called_once()
            call_args = mock_evaluate_output.call_args.kwargs
            self.assertEqual(call_args['output'], image_url)
            self.assertEqual(call_args['image_url'], image_url)

        self.loop.run_until_complete(run_test())


if __name__ == '__main__':
    unittest.main()
