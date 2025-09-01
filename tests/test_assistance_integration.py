import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import sys
import os
import tempfile
import shutil

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from supervisor_agent.core import SupervisorCore
from supervisor_agent import InterventionLevel

class TestAssistanceIntegration(unittest.TestCase):
    """
    Integration test for the stuck agent assistance feature.
    """

    def setUp(self):
        """Set up a temporary directory and a SupervisorCore instance."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.temp_dir = tempfile.mkdtemp(prefix="supervisor_assist_test_")

        # Mock the LLMManager
        mock_llm_client = MagicMock()
        # Configure the mock query to return a valid response structure
        mock_llm_client.query = AsyncMock(return_value={
            "content": {"text_response": "Mocked LLM suggestion."},
            "usage": {"input_tokens": 10, "output_tokens": 5}
        })
        mock_llm_manager = MagicMock()
        mock_llm_manager.get_client.return_value = mock_llm_client

        self.supervisor = SupervisorCore(data_dir=self.temp_dir, llm_manager=mock_llm_manager)
        self.loop.run_until_complete(self.supervisor._load_knowledge_base())

    def tearDown(self):
        """Clean up the temporary directory."""
        self.loop.close()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('supervisor_agent.core.QualityAnalyzer')
    def test_stuck_agent_assistance_flow(self, MockQualityAnalyzer):
        """
        Test that a stuck agent correctly receives a proactive ASSISTANCE intervention.
        """
        # Configure the mock to always return a very low quality score
        mock_analyzer_instance = MockQualityAnalyzer.return_value
        low_quality_metrics = unittest.mock.MagicMock()
        low_quality_metrics.confidence_score = 0.1 # Force a low score
        mock_analyzer_instance.analyze.return_value = asyncio.Future()
        mock_analyzer_instance.analyze.return_value.set_result(low_quality_metrics)

        async def run_test():
            task_id = await self.supervisor.monitor_agent(
                agent_name="stuck_agent",
                framework="test",
                task_input="a task that will cause repeated failures",
                instructions=["fail repeatedly"]
            )

            task = self.supervisor.active_tasks[task_id]
            any_output = "this output doesn't matter, the score is mocked"

            # Fail 1
            result1 = await self.supervisor.validate_output(task_id, any_output)
            self.assertTrue(result1['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 1)

            # Fail 2
            result2 = await self.supervisor.validate_output(task_id, any_output)
            self.assertTrue(result2['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 2)

            # Fail 3 - This should trigger assistance
            result3 = await self.supervisor.validate_output(task_id, any_output)
            self.assertTrue(result3['intervention_result']['intervention_required'])
            self.assertEqual(task.consecutive_failures, 3)

            final_intervention = result3['intervention_result']
            self.assertEqual(final_intervention['level'], InterventionLevel.ASSISTANCE.value)
            # We can't easily assert the reason without more mocking, so we check the level.
            self.assertIsNotNone(final_intervention['reason'])


        # Run the async test
        self.loop.run_until_complete(run_test())

if __name__ == '__main__':
    unittest.main()
