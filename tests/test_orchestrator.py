import unittest
from unittest.mock import MagicMock
import time
import sys
import os
import asyncio

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import json
from unittest.mock import MagicMock, AsyncMock, patch

from orchestrator.core import Orchestrator
from orchestrator.models import ManagedAgent, AgentStatus, ProjectGoal, TaskStatus
from supervisor_agent.core import SupervisorCore
from llm.client import LLMClient

class TestOrchestrator(unittest.TestCase):
    """Test suite for the Orchestrator."""

    def setUp(self):
        """Set up a new Orchestrator and mock dependencies for each test."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.mock_supervisor = MagicMock(spec=SupervisorCore)
        self.mock_supervisor.monitor_agent = AsyncMock()
        self.mock_supervisor.validate_output = AsyncMock()

        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.mock_llm_client.query = AsyncMock()

        self.orchestrator = Orchestrator(
            supervisor=self.mock_supervisor,
            llm_client=self.mock_llm_client,
            loop=self.loop
        )

    def tearDown(self):
        """Clean up the event loop after each test."""
        self.loop.close()

    def test_register_agent(self):
        """Test that an agent can be registered successfully."""
        self.orchestrator.register_agent("agent-1", "TestAgent", ["python"])
        self.assertEqual(len(self.orchestrator.list_agents()), 1)
        agent = self.orchestrator.get_agent("agent-1")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.name, "TestAgent")

    def test_submit_goal_llm_decomposition(self):
        """Test that the orchestrator correctly parses an LLM-generated plan."""
        # --- Setup Mock LLM Response ---
        mock_llm_response = {
            "tasks": {
                "task_1_write_code": {
                    "name": "Write the scraper code",
                    "description": "Write a Python script using BeautifulSoup to scrape the data.",
                    "required_capabilities": ["python", "file_io"],
                    "dependencies": []
                },
                "task_2_write_tests": {
                    "name": "Write unit tests",
                    "description": "Write tests for the Python scraper script.",
                    "required_capabilities": ["python", "test_execution"],
                    "dependencies": ["task_1_write_code"]
                }
            }
        }
        self.mock_llm_client.query.return_value = mock_llm_response

        # --- Execute ---
        async def run_test():
            return await self.orchestrator.submit_goal(
                "Test Scraping Project",
                "Scrape a website for data."
            )
        project = asyncio.run(run_test())

        # --- Assertions ---
        self.mock_llm_client.query.assert_called_once()
        self.assertEqual(len(project.tasks), 2)

        task1 = project.tasks["task_1_write_code"]
        task2 = project.tasks["task_2_write_tests"]

        self.assertEqual(task1.name, "Write the scraper code")
        self.assertEqual(len(task1.dependencies), 0)

        self.assertEqual(task2.name, "Write unit tests")
        self.assertIn("task_1_write_code", task2.dependencies)

    def test_get_ready_tasks(self):
        """Test the logic for identifying tasks ready for execution."""
        mock_llm_response = {
            "tasks": {
                "task1": {"name": "Task 1", "description": "", "required_capabilities": [], "dependencies": []},
                "task2": {"name": "Task 2", "description": "", "required_capabilities": [], "dependencies": ["task1"]},
                "task3": {"name": "Task 3", "description": "", "required_capabilities": [], "dependencies": ["task1"]}
            }
        }
        self.mock_llm_client.query.return_value = mock_llm_response

        project = asyncio.run(self.orchestrator.submit_goal("Test", "Test"))

        # Initially, only the task with no dependencies should be ready
        ready_tasks = project.get_ready_tasks()
        self.assertEqual(len(ready_tasks), 1)
        self.assertEqual(ready_tasks[0].name, "Write Scraper Code")

        # Mark the first task as complete
        task1_id = ready_tasks[0].task_id
        project.tasks[task1_id].status = TaskStatus.COMPLETED

        # Now, the other two tasks (which depend on the first) should be ready
        ready_tasks_after_completion = project.get_ready_tasks()
        self.assertEqual(len(ready_tasks_after_completion), 2)
        task_names = {task.name for task in ready_tasks_after_completion}
        self.assertIn("Write Unit Tests", task_names)
        self.assertIn("Generate Report", task_names)

    def test_find_available_agent_with_resource_awareness(self):
        """Test that the orchestrator selects the agent with the lowest resource load."""
        # Register three agents with the same capability
        self.orchestrator.register_agent("agent-1", "Agent One", ["python"])
        self.orchestrator.register_agent("agent-2", "Agent Two", ["python"])
        self.orchestrator.register_agent("agent-3", "Agent Three (Overloaded)", ["python"])
        self.orchestrator.register_agent("agent-4", "Agent Four (Busy)", ["python"])
        self.orchestrator.update_agent_status("agent-4", AgentStatus.BUSY)


        # Report resource usage for the idle agents
        self.orchestrator.update_agent_resources("agent-1", cpu_load=50.0, memory_load=30.0) # Total: 80
        self.orchestrator.update_agent_resources("agent-2", cpu_load=20.0, memory_load=10.0) # Total: 30 (Best)
        self.orchestrator.update_agent_resources("agent-3", cpu_load=95.0, memory_load=50.0) # Overloaded

        # Find an agent for a python task
        best_agent = self.orchestrator.find_available_agent(["python"])

        # Assert that the least loaded agent (agent-2) was chosen
        self.assertIsNotNone(best_agent)
        self.assertEqual(best_agent.agent_id, "agent-2")

    def test_find_available_agent(self):
        """Test finding an agent with the right capabilities."""
        self.orchestrator.register_agent("agent-1", "PythonAgent", ["python", "file_io"])
        self.orchestrator.register_agent("agent-2", "TextAgent", ["text_analysis"])
        self.orchestrator.register_agent("agent-3", "BusyAgent", ["python"])
        self.orchestrator.update_agent_status("agent-3", AgentStatus.BUSY)

        # Find an agent for a python task
        found_agent = self.orchestrator.find_available_agent(["python"])
        self.assertIsNotNone(found_agent)
        self.assertEqual(found_agent.agent_id, "agent-1")

        # Find an agent for a task that no idle agent can do
        found_agent_none = self.orchestrator.find_available_agent(["test_execution"])
        self.assertIsNone(found_agent_none)

        # Find an agent for the text agent
        found_text_agent = self.orchestrator.find_available_agent(["text_analysis"])
        self.assertIsNotNone(found_text_agent)
        self.assertEqual(found_text_agent.agent_id, "agent-2")

    @patch('orchestrator.core.Orchestrator._broadcast_status')
    async def test_full_execution_loop_simulation(self, mock_broadcast):
        """
        An integration-style test to simulate the orchestrator's main loop.
        """
        # --- Setup ---
        # Mock the supervisor's validate_output to return a successful result
        self.mock_supervisor.validate_output.return_value = {
            "intervention_result": {"intervention_required": False}
        }

        self.orchestrator.register_agent("agent-1", "MultiAgent", ["python", "file_io", "test_execution"])

        mock_llm_response = {
            "tasks": {
                "task_code": {"name": "Write Code", "description": "", "required_capabilities": ["python"], "dependencies": []},
                "task_test": {"name": "Write Tests", "description": "", "required_capabilities": ["test_execution"], "dependencies": ["task_code"]}
            }
        }
        self.mock_llm_client.query.return_value = mock_llm_response
        project = await self.orchestrator.submit_goal("Test Project", "Test")

        task1_id = "task_code"
        task2_id = "task_test"

        # --- Execution ---
        self.orchestrator.start()

        # Give the loop time to assign the first task
        await asyncio.sleep(0.1)

        # --- Assertions for Task 1 ---
        task1 = project.tasks[task1_id]
        agent = self.orchestrator.get_agent("agent-1")

        self.assertEqual(task1.status, TaskStatus.RUNNING)
        self.assertEqual(agent.status, AgentStatus.BUSY)
        self.assertEqual(agent.current_task_id, task1_id)

        # Let the task "finish" by sleeping past its simulated work time
        await asyncio.sleep(6)

        # --- Assertions for Task 2 ---
        # The _execute_task thread should have completed and updated the status
        self.assertEqual(task1.status, TaskStatus.COMPLETED)

        # Agent should be idle briefly before picking up the next task
        # We need to wait for the main loop to re-assign
        await asyncio.sleep(3)

        task2 = project.tasks[task2_id]
        self.assertEqual(task2.status, TaskStatus.RUNNING)
        self.assertEqual(agent.status, AgentStatus.BUSY)
        self.assertEqual(agent.current_task_id, task2_id)

        # --- Cleanup ---
        self.orchestrator.stop()

    def test_sub_orchestration_flow(self):
        """Test that a task can correctly spawn and be completed by a sub-project."""
        # --- Setup: Mock LLM to return a plan with a sub-orchestration task ---
        main_project_plan = {
            "tasks": {
                "task_simple": {"name": "A Simple Task", "description": "...", "required_capabilities": ["python"], "dependencies": []},
                "task_complex": {"name": "A Complex Task", "description": "This is the goal for the sub-project.", "required_capabilities": ["sub_orchestration"], "dependencies": ["task_simple"]}
            }
        }
        sub_project_plan = {
            "tasks": {
                "sub_task_1": {"name": "Sub Task 1", "description": "...", "required_capabilities": ["python"], "dependencies": []}
            }
        }
        # The LLM will be called twice: once for the main project, once for the sub-project.
        self.mock_llm_client.query.side_effect = [main_project_plan, sub_project_plan]

        # Mock the supervisor to always return success
        self.mock_supervisor.validate_output.return_value = {"intervention_result": {"intervention_required": False}}

        # Register a capable agent
        self.orchestrator.register_agent("agent-1", "The-Agent", ["python"])

        # --- Execution ---
        # 1. Submit the main goal
        main_project = self.loop.run_until_complete(self.orchestrator.submit_goal("Main Project", "..."))
        self.orchestrator.start()

        # 2. Let the orchestrator run to complete the simple task and create the sub-project
        time.sleep(7) # Wait for task_simple to complete

        # --- Assertions for Sub-Project Creation ---
        self.assertEqual(len(self.orchestrator.projects), 2) # Main project + sub-project
        self.assertEqual(main_project.tasks["task_simple"].status, TaskStatus.COMPLETED)
        self.assertEqual(main_project.tasks["task_complex"].status, TaskStatus.WAITING_ON_SUB_PROJECT)

        sub_project_id = main_project.tasks["task_complex"].output_data["sub_project_id"]
        self.assertIn(sub_project_id, main_project.sub_goal_ids)

        sub_project = self.orchestrator.projects[sub_project_id]
        self.assertEqual(sub_project.parent_goal_id, main_project.goal_id)

        # 3. Manually mark the sub-project as complete to test the final part of the loop
        with self.orchestrator._lock:
            sub_project.status = "COMPLETED"

        # 4. Let the orchestrator run again to detect the completion
        time.sleep(3)

        # --- Final Assertions ---
        self.assertEqual(main_project.tasks["task_complex"].status, TaskStatus.COMPLETED)

        # --- Cleanup ---
        self.orchestrator.stop()


if __name__ == '__main__':
    unittest.main()
