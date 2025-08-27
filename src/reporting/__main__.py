#!/usr/bin/env python3
"""
Main entry point for the Comprehensive Reporting and Feedback System
"""

import sys
import argparse
import json
from pathlib import Path
import asyncio

from ..supervisor.integrated_supervisor import IntegratedSupervisor, SupervisorConfig
from .integrated_system import IntegratedReportingSystem, IntegratedReportingConfig

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Comprehensive Reporting and Feedback System for Supervisor Agent'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON)'
    )

    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo scenario'
    )

    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='Generate comprehensive report and exit'
    )

    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Start dashboard only (no background processing)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='reporting_output',
        help='Output directory for reports and logs'
    )

    parser.add_argument(
        '--port', '-p',
        type=int,
        default=5000,
        help='Dashboard port (default: 5000)'
    )

    args = parser.parse_args()

    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_data = json.load(f)
        reporting_config = IntegratedReportingConfig(**config_data)
    else:
        reporting_config = IntegratedReportingConfig(
            base_output_dir=args.output_dir,
            dashboard_port=args.port,
            background_processing=not args.dashboard_only
        )

    # This is the main async block for running the server
    async def run_systems():
        reporting_system = IntegratedReportingSystem(reporting_config)
        supervisor_config = SupervisorConfig()
        supervisor = IntegratedSupervisor(reporting_system, supervisor_config)

        if args.demo:
            print("Running demo scenario...")
            await supervisor.start() # Start the supervisor and its subsystems

            # The old demo scenario is deprecated, log some simple events for testing
            reporting_system.log_event('demo_start', 'main', 'Demo started')
            await asyncio.sleep(2)
            reporting_system.log_event('demo_end', 'main', 'Demo finished')

            # Keep alive for testing, then stop
            await asyncio.sleep(15)
            await supervisor.stop()
            return

        if args.report:
            print("Generating comprehensive report...")
            # This needs to be made async or run in an executor
            # For now, we assume it's a quick operation
            reports = reporting_system.generate_comprehensive_report()
            print(f"Reports generated: {reports}")
            return

        print("Starting Comprehensive Supervisor System...")
        print(f"Dashboard: http://localhost:{reporting_config.dashboard_port}")
        print(f"Output directory: {reporting_config.base_output_dir}")
        print("Press Ctrl+C to stop")

        await supervisor.start()
        while True:
            await asyncio.sleep(1)

    supervisor_task = None
    try:
        supervisor_task = asyncio.run(run_systems())
    except KeyboardInterrupt:
        print("\nStopping system...")
        if supervisor_task:
            # This is not perfect, as supervisor is not in this scope
            # For now, Ctrl+C will kill the process
            pass
        print("System stopped.")


if __name__ == '__main__':
    main()
