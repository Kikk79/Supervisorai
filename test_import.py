import asyncio
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestImport(unittest.IsolatedAsyncioTestCase):
    async def test_import_integrated_supervisor(self):
        try:
            from supervisor.integrated_supervisor import IntegratedSupervisor, SupervisorConfig
            config = SupervisorConfig()
            supervisor = IntegratedSupervisor(config=config, reporting_system=None)
            self.assertIsNotNone(supervisor)
        except Exception as e:
            self.fail(f"Importing or instantiating IntegratedSupervisor failed with: {e}")

if __name__ == '__main__':
    unittest.main()
