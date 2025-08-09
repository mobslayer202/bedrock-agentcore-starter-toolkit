"""AWS X-Ray service for observability setup."""

import logging
from typing import Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

log = logging.getLogger(__name__)


def validate_aws_credentials(session: Optional[boto3.Session] = None) -> bool:
    """Check if AWS credentials are available and valid.

    Args:
        session: Optional boto3 session to use

    Returns:
        bool: True if AWS credentials are available and valid
    """
    try:
        if session is None:
            session = boto3.Session()

        sts_client = session.client("sts")
        sts_client.get_caller_identity()
        return True
    except (NoCredentialsError, ClientError, BotoCoreError):
        return False


def is_xray_transaction_search_configured(session: Optional[boto3.Session] = None) -> bool:
    """Check if X-Ray Transaction Search is already configured.

    Args:
        session: Optional boto3 session to use

    Returns:
        bool: True if already configured for CloudWatch Logs
    """
    try:
        if session is None:
            session = boto3.Session()

        xray_client = session.client("xray")
        response = xray_client.get_trace_segment_destination()

        # Check if destination is set to CloudWatchLogs
        return response.get("Destination") == "CloudWatchLogs"

    except (ClientError, BotoCoreError):
        # If we can't check the configuration, assume it's not configured
        return False


def setup_xray_transaction_search(session: Optional[boto3.Session] = None) -> Tuple[bool, str]:
    """Set up X-Ray Transaction Search for cost-effective trace collection.

    Performs the equivalent of these AWS CLI commands:
    1. aws xray update-trace-segment-destination --destination CloudWatchLogs
    2. aws xray update-indexing-rule --name "Default" --rule '{"Probabilistic": {"DesiredSamplingPercentage": 1}}'

    Args:
        session: Optional boto3 session to use

    Returns:
        Tuple[bool, str]: (success, message)
    """
    try:
        # First, validate AWS credentials are available
        if not validate_aws_credentials(session):
            return (
                False,
                "AWS credentials not available or not configured. Please ensure AWS credentials are configured.",
            )

        if session is None:
            session = boto3.Session()

        # Check if already configured
        if is_xray_transaction_search_configured(session):
            log.info("X-Ray Transaction Search already configured")
            return True, "X-Ray Transaction Search already configured"

        log.info("Setting up X-Ray Transaction Search...")
        xray_client = session.client("xray")

        # Step 1: Update trace segment destination to CloudWatchLogs
        log.debug("Updating X-Ray trace segment destination to CloudWatchLogs")
        try:
            xray_client.update_trace_segment_destination(Destination="CloudWatchLogs")
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            if error_code == "AccessDeniedException":
                return False, f"Insufficient permissions for X-Ray setup: {error_msg}"
            elif error_code == "ValidationException":
                return False, f"Invalid X-Ray configuration: {error_msg}"
            else:
                return False, f"Failed to update trace segment destination: {error_msg}"

        # Step 2: Update indexing rule for 1% sampling
        log.debug("Updating X-Ray indexing rule with 1% sampling")
        try:
            xray_client.update_indexing_rule(Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 1}})
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_msg = e.response["Error"]["Message"]
            if error_code == "AccessDeniedException":
                return False, f"Insufficient permissions for X-Ray indexing rule update: {error_msg}"
            elif error_code == "ValidationException":
                return False, f"Invalid indexing rule configuration: {error_msg}"
            else:
                return False, f"Failed to update indexing rule: {error_msg}"

        log.info("X-Ray Transaction Search configured successfully")
        return True, "X-Ray Transaction Search configured successfully"

    except BotoCoreError as e:
        error_msg = f"AWS service error configuring X-Ray Transaction Search: {e}"
        log.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error configuring X-Ray Transaction Search: {e}"
        log.error(error_msg)
        return False, error_msg


def get_xray_permissions_help() -> str:
    """Get help text for required X-Ray permissions.

    Returns:
        str: Help text with required IAM permissions
    """
    return """
Required IAM permissions for X-Ray Transaction Search setup:
- xray:UpdateTraceSegmentDestination
- xray:GetTraceSegmentDestination
- xray:UpdateIndexingRule
- xray:GetIndexingRules

You can add these permissions to your IAM user or role, or use a policy like:
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "xray:UpdateTraceSegmentDestination",
                "xray:GetTraceSegmentDestination",
                "xray:UpdateIndexingRule",
                "xray:GetIndexingRules"
            ],
            "Resource": "*"
        }
    ]
}
"""
