"""Utility functions for agent log information."""

from typing import Optional, Tuple


def get_agent_log_paths(agent_id: str, endpoint_name: Optional[str] = None) -> Tuple[str, str]:
    """Get CloudWatch log group paths for an agent.

    Args:
        agent_id: The agent ID
        endpoint_name: The endpoint name (defaults to "DEFAULT")

    Returns:
        Tuple of (runtime_log_group, otel_log_group)
    """
    endpoint_name = endpoint_name or "DEFAULT"
    runtime_log_group = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint_name}"
    otel_log_group = f"/aws/bedrock-agentcore/runtimes/{agent_id}-{endpoint_name}/otel-rt-logs"
    return runtime_log_group, otel_log_group


def get_aws_tail_commands(log_group: str) -> tuple[str, str, str, str, str, str]:
    """Get AWS CLI tail commands for a log group.

    Args:
        log_group: The CloudWatch log group path

    Returns:
        Tuple of (base_follow_cmd, base_since_cmd, container_follow_cmd, container_since_cmd, otel_follow_cmd, otel_since_cmd)
    """
    base_follow_cmd = f"aws logs tail {log_group} --follow"
    base_since_cmd = f"aws logs tail {log_group} --since 1h"
    
    container_follow_cmd = f"{base_follow_cmd} | grep ' runtime-logs '"
    container_since_cmd = f"{base_since_cmd} | grep ' runtime-logs '"
    
    otel_follow_cmd = f"{base_follow_cmd} | grep ' otel-rt-logs '"
    otel_since_cmd = f"{base_since_cmd} | grep ' otel-rt-logs '"
    
    return base_follow_cmd, base_since_cmd, container_follow_cmd, container_since_cmd, otel_follow_cmd, otel_since_cmd 
