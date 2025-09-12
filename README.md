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

*   **Code-Aware Supervision:**
    *   The supervisor can now understand code quality. When an agent produces Python code, the system uses the **`pylint` static analysis tool** (`analysis/code_analyzer.py`) to check for errors, code smells, and style issues.
    *   The number of errors found is factored directly into the `AgentState` passed to the Expectimax agent, making its decisions about code much more intelligent.

*   **Multi-modal Supervision:**
    *   The supervisor now has a new sense: vision. It can evaluate image-based outputs from agents.
    *   When an agent's output is an image URL, the `LLMJudge` uses a vision-capable model (e.g., Claude 3 Opus) to evaluate the image against the task goals.

*   **Feedback-Driven Learning:**
    *   The supervisor can **learn from user feedback**. The dashboard allows a human to correct a bad decision, and this feedback is used to retrain the weights of the Expectimax agent's evaluation function via `supervisor_agent/feedback_trainer.py`.
    *   This creates a powerful self-improvement loop, allowing the supervisor's judgment to get better over time.

### **Orchestration Engine**

*   **Autonomous Orchestrator with Multi-LLM Support:**
    *   Manages a pool of specialized agents with different capabilities.
    *   Features an **LLM-powered task planner**. The system is architected to use multiple LLM providers concurrently (e.g., Anthropic, OpenAI), loading its configuration from `config/llm_config.json`.
    *   Different models can be used for different tasks (e.g., a fast model for planning, a powerful model for judging) to optimize for cost and performance.

*   **Sub-Orchestration:**
    *   For extremely complex goals, the main orchestrator can now delegate tasks to **sub-projects**. The LLM planner is instructed to identify tasks that are themselves large projects and assign them a `sub_orchestration` capability.
    *   The orchestrator then creates a new, nested `ProjectGoal` and monitors it, allowing for hierarchical, recursive problem-solving.

*   **Resource-Aware Task Assignment:**
    *   The orchestrator is now aware of agent system resources. Agents can report their CPU and memory load via a new API endpoint.
    *   The `find_available_agent` logic has been enhanced to filter out agents with high resource usage (e.g., >90%) and to prioritize assigning tasks to the least-loaded agent available.

*   **The Agent Factory:**
    *   The system is capable of building new agents for itself. When a "build agent" goal is submitted, the orchestrator creates a sub-project to manage a team of coding agents that write, test, and register a new agent into the pool.

### **Assistance & UI**

*   **Proactive Research Assistant:**
    *   The supervisor can detect when an agent is "stuck". It then autonomously formulates a search query, uses **Google Search** to find relevant help articles, and uses an **LLM to synthesize a helpful suggestion**.

*   **Cost Analysis:**
    *   A `CostTracker` service logs every LLM call made by the system. It uses model-specific pricing to calculate the cost of each call and provides a detailed report, which can be viewed on the dashboard.

*   **Interactive Dashboard with Real-Time Updates:**
    *   A comprehensive web dashboard (`examples/dashboard.html`) serves as the central UI.
    *   The dashboard now features **real-time log and status streaming** via a dedicated WebSocket connection, making the UI highly responsive.
    *   It includes an **interactive debugger** that visualizes the Expectimax agent's entire decision tree as a flowchart.

## 3. System Architecture

The project is organized into a standard Python project structure:

*   `src/supervisor_agent/`: Contains the core `SupervisorCore` and the `ExpectimaxAgent`.
*   `src/orchestrator/`: Contains the `Orchestrator` and its data models.
*   `src/researcher/`: Contains the `ResearchAssistor`.
*   `src/analysis/`: Contains the `CodeQualityAnalyzer`.
*   `src/agent_factory/`: Contains templates for building new agents.
*   `src/llm/`: Contains the multi-LLM client architecture (`base.py`, `manager.py`, etc.).
*   `src/server/`: Contains the main ASGI server (`main.py`).
*   `examples/dashboard.html`: The all-in-one web interface.
*   `tests/`: Contains unit and integration tests.

## 4. Setup and Installation

To get the project running, follow these steps:

1.  **Set up a Python virtual environment:**
    *   This project uses Python's standard `venv` module.
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
    pip install -r requirements.txt
    ```

4.  **Configure API Keys (Optional):**
    *   Create a `.env` file in the root directory or set environment variables for the LLM providers you wish to use.
    ```
    ANTHROPIC_API_KEY="your-anthropic-key"
    OPENAI_API_KEY="your-openai-key"
    ```
    *   If API keys are not set, the relevant clients will return mocked responses.

## 5. How to Run the System

1.  **Start the Server:**
    *   The application is an ASGI web server and should be run with `uvicorn`. The following command also sets the `PYTHONPATH` correctly, which is required for the application's imports to work.
    ```bash
    PYTHONPATH=$(pwd)/src .venv/bin/uvicorn src.server.main:mcp --port 8765
    ```
    *   You should see output from `uvicorn` indicating the server is running on `http://127.0.0.1:8765`.

2.  **Use the Dashboard:**
    *   Open the `examples/dashboard.html` file in your web browser. This file is self-contained and will connect to the local server automatically.

## 6. Future Roadmap

This project has a rich roadmap for future development.

### Planned Features
*   **Interactive Goal Definition:** Create a UI for drag-and-drop task planning.
*   **Authentication & Multi-User:** Add a proper user login system.

### Next Frontier Ideas
*   **The Ethics Guardian:** A specialized supervisor to enforce an "ethical constitution."
*   **The Self-Improving Supervisor (Meta-Learning):** A supervisor that learns from its own interventions to improve its policies.
*   **Predictive Intervention Engine:** A system that analyzes an agent's work in real-time to predict and prevent failures.
*   **Full System Autonomy:** Connect the orchestrator to external data streams to allow it to discover and propose its own goals.
*   **Meta-Supervision:** A system that can analyze its own performance and autonomously refactor its own source code or prompts.
*   **Human-AI Symbiosis:** Evolve the UI into a true collaborative partner with conversational planning and deeply explainable AI (XAI).
*   **Decentralized Swarm Orchestration:** Move from a single orchestrator to a decentralized swarm of orchestrators.
*   **Embodied AI & Physical World Control:** Connect the orchestrator to physical hardware (robotics, IoT).

## 7. Credits

This project was developed as part of the Minimax Agent Hackathon.

**Lead AI Software Engineer:** Jules
