# Programmatic Usage of Import Agent CLI

This document explains how to use the `import_agent` command programmatically without requiring user input via questionary.

## Overview

The `import_agent` command has been enhanced to accept command-line flags for all choices that were previously made via interactive prompts. This allows you to run the command programmatically in scripts or automation workflows.

## Available Flags

| Flag | Description | Type | Default |
|------|-------------|------|---------|
| `--agent-id` | ID of the Bedrock Agent to import | string | None |
| `--agent-alias-id` | ID of the Agent Alias to use | string | None |
| `--target-platform` | Target platform (langchain or strands) | string | None |
| `--debug` | Enable debug mode | boolean | False |
| `--verbose` | Enable verbose mode | boolean | False |
| `--disable-memory` | Disable AgentCore Memory primitive | boolean | False |
| `--disable-code-interpreter` | Disable AgentCore Code Interpreter primitive | boolean | False |
| `--disable-observability` | Disable AgentCore Observability primitive | boolean | False |
| `--deploy-runtime` | Deploy to AgentCore Runtime | boolean | False |
| `--run-option` | How to run the agent (locally, runtime, none) | string | None |
| `--output-dir` | Output directory for generated code | string | "./output/" |

## Example Usage

### Basic Usage

```bash
python -m src.bedrock_agentcore_starter_toolkit.cli.cli import-agent \
  --agent-id "YOUR_AGENT_ID" \
  --agent-alias-id "YOUR_AGENT_ALIAS_ID" \
  --target-platform "langchain" \
  --run-option "none"
```

### With All Options

```bash
python -m src.bedrock_agentcore_starter_toolkit.cli.cli import-agent \
  --agent-id "YOUR_AGENT_ID" \
  --agent-alias-id "YOUR_AGENT_ALIAS_ID" \
  --target-platform "langchain" \
  --verbose \
  --output-dir "./custom_output/" \
  --run-option "none"
```

Note: Memory, Code Interpreter, and Observability primitives are enabled by default.

### Deploy to AgentCore Runtime

```bash
python -m src.bedrock_agentcore_starter_toolkit.cli.cli import-agent \
  --agent-id "YOUR_AGENT_ID" \
  --agent-alias-id "YOUR_AGENT_ALIAS_ID" \
  --target-platform "strands" \
  --deploy-runtime \
  --run-option "runtime"
```

## Behavior

- If required flags like `--agent-id`, `--agent-alias-id`, or `--target-platform` are not provided, the command will fall back to interactive prompts.
- Boolean flags like `--verbose`, `--debug`, `--disable-memory`, etc. don't require values; their presence sets them to `True`.
- If neither `--verbose` nor `--debug` flags are provided, the command will prompt the user to enable verbose mode.
- Either `--verbose` or `--debug` will enable verbose mode. Use `--verbose` for standard verbose output and `--debug` for more detailed debugging information.
- Memory, Code Interpreter, and Observability primitives are enabled by default. Use `--disable-memory`, `--disable-code-interpreter`, or `--disable-observability` to disable them.
- If the `--deploy-runtime` flag is not provided, the command will prompt the user whether to deploy the agent to AgentCore Runtime.
- If the `--run-option` flag is not provided, the command will prompt the user to select how to run the agent.
- The `--run-option` can be one of:
  - `locally`: Run the agent locally
  - `runtime`: Run on AgentCore Runtime (requires `--deploy-runtime`)
  - `none`: Don't run the agent

## Test Script

A test script `test_import_agent_cli.py` is provided to demonstrate how to use the command programmatically:

```bash
python test_import_agent_cli.py
```

This script shows example commands but doesn't actually run them. To run the commands, uncomment the `subprocess.run(command)` line in the script and replace the placeholder agent IDs with your actual IDs.
