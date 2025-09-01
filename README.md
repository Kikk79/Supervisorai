# AI Supervisor & Orchestrator
### *A Submission for the Minimax Agent Hackathon*

---

## 1. Project Overview

This project is a sophisticated, AI-powered system designed to supervise, manage, and assist other AI agents. It has evolved from a simple monitoring script into a multi-layered platform with advanced capabilities for intelligent oversight and autonomous operation.

The system is built around three core concepts:
*   **Supervision:** A supervisor agent that uses a probabilistic model (Expectimax) to watch over a working agent, predict potential issues, and intervene when necessary.
*   **Orchestration:** An autonomous orchestrator that can manage a pool of specialized agents, decompose high-level goals into a dependency graph of tasks, and manage the entire execution workflow, including delegating complex tasks to sub-orchestrators.
*   **Assistance:** A proactive research assistant that can detect when an agent is stuck, perform web searches to find solutions for its errors, and provide intelligent suggestions to help it recover.

## 2. Core Features

This project includes a rich set of features, demonstrating a robust and intelligent architecture.

### **Supervision Engine**

*   **Intelligent Supervisor Agent:**
    *   Uses an **Expectimax algorithm** (`supervisor_agent/expectimax_agent.py`) to make nuanced decisions about whether to `ALLOW`, `WARN`, `CORRECT`, or `ESCALATE` an agent's output. This is not based on simple rules, but on a probabilistic model of future outcomes.
    *   The decision-making is based on a weighted evaluation of the agent's state, including output quality, task drift, error count, and resource usage.

*   **Code-Aware Supervision (New!):**
    *   The supervisor can now understand code quality. When an agent produces Python code, the system uses the **`pylint` static analysis tool** (`analysis/code_analyzer.py`) to check for errors, code smells, and style issues.
    *   The number of errors found is factored directly into the `AgentState` passed to the Expectimax agent, making its decisions about code much more intelligent.

*   **Feedback-Driven Learning:**
    *   The supervisor can **learn from user feedback**. The dashboard allows a human to correct a bad decision, and this feedback is used to retrain the weights of the Expectimax agent's evaluation function via `supervisor_agent/feedback_trainer.py`.
    *   This creates a powerful self-improvement loop, allowing the supervisor's judgment to get better over time.

### **Orchestration Engine**

*   **Autonomous Orchestrator:**
    *   Manages a pool of specialized agents with different capabilities.
    *   Features an **LLM-powered task planner** that can take a high-level goal (e.g., "build a web scraper") and autonomously decompose it into a detailed, multi-step plan with dependencies (`orchestrator/prompts.py`).

*   **Sub-Orchestration (New!):**
    *   For extremely complex goals, the main orchestrator can now delegate tasks to **sub-projects**. The LLM planner is instructed to identify tasks that are themselves large projects and assign them a `sub_orchestration` capability.
    *   The orchestrator then creates a new, nested `ProjectGoal` and monitors it, allowing for hierarchical, recursive problem-solving.

*   **Resource-Aware Task Assignment (New!):**
    *   The orchestrator is now aware of agent system resources. Agents can report their CPU and memory load via a new API endpoint.
    *   The `find_available_agent` logic has been enhanced to filter out agents with high resource usage (e.g., >90%) and to prioritize assigning tasks to the least-loaded agent available.

### **Assistance & UI**

*   **Proactive Research Assistant:**
    *   The supervisor can detect when an agent is "stuck" (e.g., failing repeatedly).
    *   It then autonomously formulates a search query, uses **Google Search** to find relevant help articles, reads the content, and uses an **LLM to synthesize a helpful suggestion**.
    *   This allows the system to solve its own problems without human intervention.

*   **Interactive Dashboard with Real-Time Updates (New!):**
    *   A comprehensive web dashboard (`examples/dashboard.html`) serves as the central UI.
    *   The dashboard now features **real-time log and status streaming** via a dedicated WebSocket connection. The inefficient polling mechanism has been removed, making the UI highly responsive.
    *   It includes an **interactive debugger** that visualizes the Expectimax agent's entire decision tree as a flowchart, allowing for deep "what-if" analysis.

## 3. System Architecture

The project is organized into a standard Python project structure:

*   `src/supervisor_agent/`: Contains the core `SupervisorCore` and the `ExpectimaxAgent`.
*   `src/orchestrator/`: Contains the `Orchestrator` and its data models for managing projects and tasks.
*   `src/researcher/`: Contains the `ResearchAssistor` responsible for proactive help.
*   `src/analysis/`: Contains the new `CodeQualityAnalyzer`.
*   `src/llm/`: Contains the generic `LLMClient` for interacting with language models.
*   `src/server/`: Contains the server (`main.py`) that exposes all functionality through a WebSocket-based API.
*   `examples/dashboard.html`: The all-in-one web interface for interacting with the system.
*   `tests/`: Contains unit and integration tests for the various components.

## 4. Setup and Installation

To get the project running, follow these steps:

1.  **Set up a Python virtual environment:**
    *   This project uses Python's standard `venv` module. Do not use `uv`, as it was found to be unreliable in some environments.
    ```bash
    python3 -m venv .venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    *   Install all required packages using the `pip` from your new virtual environment.
    ```bash
    .venv/bin/pip install -r requirements.txt
    ```

4.  **Configure API Keys (Optional):**
    *   The system uses the Anthropic Claude 3 API for its LLM capabilities. To enable this, set the following environment variable:
    ```bash
    export ANTHROPIC_API_KEY="your-api-key-here"
    ```
    *   If the API key is not set, the LLM-powered features will return mocked responses, but the rest of the system will still be functional.

## 5. How to Run the System

1.  **Start the Server:**
    *   The application is an ASGI web server and should be run with `uvicorn`. The following command also sets the `PYTHONPATH` correctly, which is required for the application's imports to work.
    ```bash
    PYTHONPATH=$(pwd)/src .venv/bin/uvicorn src.server.main:mcp --port 8765
    ```
    *   You should see output from `uvicorn` indicating the server is running on `http://127.0.0.1:8765`.

2.  **Use the Dashboard:**
    *   Open the `examples/dashboard.html` file in your web browser. This file is self-contained and will connect to the local server automatically.

## 6. Future Roadmap (Planned Features)

The following features are part of the project's future roadmap and have not yet been implemented:

*   **Multi-modal Supervision:** Enhance the `LLMJudge` to use vision models to evaluate the quality of non-text output like images.
*   **Cost Analysis:** Implement a `CostTracker` to monitor token usage and provide reports on the operational cost of the system.
*   **Interactive Goal Definition:** Create a more advanced UI that allows users to drag-and-drop tasks to create or modify the orchestrator's plans.
*   **Authentication & Multi-User:** Add a proper user login system so that different people can manage their own agents and projects.

## 7. Credits

This project was developed as part of the Minimax Agent Hackathon.

**Lead AI Software Engineer:** Jules
