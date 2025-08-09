"""Tests for Bedrock AgentCore launch operation."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from bedrock_agentcore_starter_toolkit.operations.runtime.launch import (
    _ensure_execution_role,
    launch_bedrock_agentcore,
)
from bedrock_agentcore_starter_toolkit.services.xray import (
    is_xray_transaction_search_configured,
    setup_xray_transaction_search,
    validate_aws_credentials,
)
from bedrock_agentcore_starter_toolkit.utils.runtime.config import save_config
from bedrock_agentcore_starter_toolkit.utils.runtime.schema import (
    AWSConfig,
    BedrockAgentCoreAgentSchema,
    BedrockAgentCoreConfigSchema,
    BedrockAgentCoreDeploymentInfo,
    NetworkConfiguration,
    ObservabilityConfig,
)


# Test Helper Functions
def create_test_config(
    tmp_path,
    agent_name="test-agent",
    entrypoint="test_agent.py",
    region="us-west-2",
    account="123456789012",
    execution_role=None,
    execution_role_auto_create=False,
    ecr_repository=None,
    ecr_auto_create=False,
    agent_id=None,
    agent_session_id=None,
):
    """Create a test configuration with customizable parameters."""
    config_path = tmp_path / ".bedrock_agentcore.yaml"
    agent_config = BedrockAgentCoreAgentSchema(
        name=agent_name,
        entrypoint=entrypoint,
        container_runtime="docker",
        aws=AWSConfig(
            region=region,
            account=account,
            execution_role=execution_role,
            execution_role_auto_create=execution_role_auto_create,
            ecr_repository=ecr_repository,
            ecr_auto_create=ecr_auto_create,
            network_configuration=NetworkConfiguration(),
            observability=ObservabilityConfig(),
        ),
        bedrock_agentcore=BedrockAgentCoreDeploymentInfo(
            agent_id=agent_id,
            agent_session_id=agent_session_id,
        ),
    )
    project_config = BedrockAgentCoreConfigSchema(default_agent=agent_name, agents={agent_name: agent_config})
    save_config(project_config, config_path)
    return config_path


def create_test_agent_file(tmp_path, filename="test_agent.py", content="# test agent"):
    """Create a test agent file."""
    agent_file = tmp_path / filename
    agent_file.write_text(content)
    return agent_file


class MockAWSClientFactory:
    """Factory for creating consistent AWS client mocks."""

    def __init__(self, account="123456789012", region="us-west-2"):
        self.account = account
        self.region = region
        self._setup_clients()

    def _setup_clients(self):
        """Setup all AWS service client mocks."""
        # IAM Client Mock
        self.iam_client = MagicMock()
        self.iam_client.get_role.return_value = {
            "Role": {
                "Arn": f"arn:aws:iam::{self.account}:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # CodeBuild Client Mock
        self.codebuild_client = MagicMock()
        self.codebuild_client.batch_get_builds.return_value = {
            "builds": [{"buildStatus": "SUCCEEDED", "currentPhase": "COMPLETED"}]
        }
        self.codebuild_client.create_project.return_value = {}
        self.codebuild_client.start_build.return_value = {"build": {"id": "build-123"}}

        # S3 Client Mock
        self.s3_client = MagicMock()
        self.s3_client.head_bucket.return_value = {}
        self.s3_client.upload_file.return_value = {}

        # STS Client Mock
        self.sts_client = MagicMock()
        self.sts_client.get_caller_identity.return_value = {"Account": self.account}

    def get_client(self, service_name):
        """Get a mock client for the specified service."""
        clients = {
            "iam": self.iam_client,
            "codebuild": self.codebuild_client,
            "s3": self.s3_client,
            "sts": self.sts_client,
        }
        return clients.get(service_name, MagicMock())

    def setup_full_session_mock(self, mock_boto3_clients):
        """Setup the complete session mock with all AWS clients."""
        mock_session = mock_boto3_clients["session"]
        mock_session.client.side_effect = self.get_client
        mock_session.region_name = self.region

    def setup_session_mock(self, mock_boto3_clients):
        """Setup the session mock to use our client factory (legacy method)."""
        self.setup_full_session_mock(mock_boto3_clients)


def assert_codebuild_workflow_called(mock_factory):
    """Assert that CodeBuild workflow was properly executed."""
    mock_factory.codebuild_client.create_project.assert_called()
    mock_factory.codebuild_client.start_build.assert_called()
    mock_factory.codebuild_client.batch_get_builds.assert_called()


def assert_config_updated_with_role(config_path, expected_role_arn):
    """Assert that config was updated with the expected execution role."""
    from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

    updated_config = load_config(config_path)
    updated_agent = list(updated_config.agents.values())[0]
    assert updated_agent.aws.execution_role == expected_role_arn
    assert updated_agent.aws.execution_role_auto_create is False


def assert_no_agent_deployment_calls(mock_boto3_clients):
    """Assert that no agent deployment calls were made (for ECR-only tests)."""
    mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.assert_not_called()
    mock_boto3_clients["bedrock_agentcore"].update_agent_runtime.assert_not_called()


class TestLaunchBedrockAgentCore:
    """Test launch_bedrock_agentcore functionality."""

    def test_launch_local_mode(self, mock_container_runtime, tmp_path):
        """Test local deployment."""
        config_path = create_test_config(tmp_path)
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        with patch(
            "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
            return_value=mock_container_runtime,
        ):
            result = launch_bedrock_agentcore(config_path, local=True)

        # Verify local mode result
        assert result.mode == "local"
        assert result.tag == "bedrock_agentcore-test-agent:latest"
        assert result.port == 8080
        assert hasattr(result, "runtime")
        mock_container_runtime.build.assert_called_once()

    def test_launch_cloud_with_ecr_auto_create(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test cloud deployment with ECR creation."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_auto_create=True,
        )
        create_test_agent_file(tmp_path)

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository") as mock_create_ecr,
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
            ) as mock_create_role,
        ):
            mock_create_ecr.return_value = "123456789012.dkr.ecr.us-west-2.amazonaws.com/bedrock_agentcore-test-agent"
            mock_create_role.return_value = "arn:aws:iam::123456789012:role/TestRole"

            result = launch_bedrock_agentcore(config_path, local=False)

            # Verify codebuild mode result
            assert result.mode == "codebuild"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")
            assert hasattr(result, "ecr_uri")
            assert hasattr(result, "codebuild_id")

            # Verify CodeBuild workflow was executed
            assert_codebuild_workflow_called(mock_factory)

    def test_launch_cloud_existing_agent(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test updating existing agent."""
        config_path = create_test_config(
            tmp_path,
            account="023456789012",
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
            agent_id="existing-agent-id",
        )
        create_test_agent_file(tmp_path)

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory(account="023456789012")
        mock_factory.setup_session_mock(mock_boto3_clients)

        with patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository"):
            result = launch_bedrock_agentcore(config_path, local=False)

            # Verify update was called (not create)
            mock_boto3_clients["bedrock_agentcore"].update_agent_runtime.assert_called_once()
            assert result.mode == "codebuild"

    def test_launch_build_failure(self, mock_container_runtime, tmp_path):
        """Test error handling for build failures."""
        config_path = create_test_config(tmp_path)
        create_test_agent_file(tmp_path)

        # Mock build failure
        mock_container_runtime.build.return_value = (False, ["Error: build failed", "Missing dependency"])

        with patch(
            "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
            return_value=mock_container_runtime,
        ):
            with pytest.raises(RuntimeError, match="Build failed"):
                launch_bedrock_agentcore(config_path, local=True)

    def test_launch_missing_config(self, tmp_path):
        """Test error when config file not found."""
        nonexistent_config = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError):
            launch_bedrock_agentcore(nonexistent_config)

    def test_launch_invalid_config(self, tmp_path):
        """Test validation errors."""
        config_path = create_test_config(tmp_path, entrypoint="")  # Invalid empty entrypoint
        create_test_agent_file(tmp_path)

        with pytest.raises(ValueError, match="Invalid configuration"):
            launch_bedrock_agentcore(config_path, local=False)

    def test_launch_local_build_cloud_deployment(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test local build with cloud deployment (use_codebuild=False)."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }
        mock_boto3_clients["session"].client.return_value = mock_iam_client

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            result = launch_bedrock_agentcore(config_path, local=False, use_codebuild=False)

            # Verify local build with cloud deployment
            assert result.mode == "cloud"
            assert result.tag == "bedrock_agentcore-test-agent:latest"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")
            assert hasattr(result, "ecr_uri")
            assert hasattr(result, "build_output")

            # Verify local build was used (not CodeBuild)
            mock_container_runtime.build.assert_called_once()
            mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.assert_called_once()

    def test_launch_missing_ecr_repository(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test error when ECR repository not configured."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_auto_create=False,  # No auto-create and no ECR repository
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }
        mock_boto3_clients["session"].client.return_value = mock_iam_client

        with patch(
            "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
            return_value=mock_container_runtime,
        ):
            with pytest.raises(ValueError, match="ECR repository not configured"):
                launch_bedrock_agentcore(config_path, local=False)

    def test_launch_cloud_with_execution_role_auto_create(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test cloud deployment with execution role auto-creation."""
        config_path = create_test_config(
            tmp_path,
            execution_role_auto_create=True,  # Enable auto-creation
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Role name will use random suffix, so we can't predict the exact name
        created_role_arn = "arn:aws:iam::123456789012:role/AmazonBedrockAgentCoreSDKRuntime-us-west-2-abc123xyz9"

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_full_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
            ) as mock_get_or_create_role,
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            mock_get_or_create_role.return_value = created_role_arn

            result = launch_bedrock_agentcore(config_path, local=False, use_codebuild=False)

            # Verify execution role creation was called
            mock_get_or_create_role.assert_called_once()

            # Verify role creation parameters
            call_args = mock_get_or_create_role.call_args
            assert call_args.kwargs["region"] == "us-west-2"
            assert call_args.kwargs["account_id"] == "123456789012"
            assert call_args.kwargs["agent_name"] == "test-agent"

            # Verify cloud deployment succeeded
            assert result.mode == "cloud"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")

        # Verify the config was updated with the created role
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        updated_config = load_config(config_path)
        updated_agent = updated_config.agents["test-agent"]
        assert updated_agent.aws.execution_role == created_role_arn
        assert updated_agent.aws.execution_role_auto_create is False  # Should be disabled after creation

    def test_launch_cloud_with_existing_execution_role(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test cloud deployment with existing execution role (no auto-creation)."""
        existing_role_arn = "arn:aws:iam::123456789012:role/existing-test-role"

        config_path = create_test_config(
            tmp_path,
            execution_role=existing_role_arn,
            execution_role_auto_create=True,  # Should be ignored since role exists
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": existing_role_arn,
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }
        mock_boto3_clients["session"].client.return_value = mock_iam_client

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
            ) as mock_create_role,
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            result = launch_bedrock_agentcore(config_path, local=False, use_codebuild=False)

            # Verify execution role creation was NOT called (role already exists)
            mock_create_role.assert_not_called()

            # Verify cloud deployment succeeded
            assert result.mode == "cloud"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")

        # Verify the config was not modified (role already existed)
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        updated_config = load_config(config_path)
        updated_agent = updated_config.agents["test-agent"]
        assert updated_agent.aws.execution_role == existing_role_arn

    def test_launch_missing_execution_role_no_auto_create(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test error when execution role not configured and auto-create disabled."""
        config_path = create_test_config(
            tmp_path,
            execution_role_auto_create=False,  # No auto-create and no execution role
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            with pytest.raises(ValueError, match="Missing 'aws.execution_role' for cloud deployment"):
                launch_bedrock_agentcore(config_path, local=False)

    def test_launch_cloud_conflict_exception_graceful_handling(
        self, mock_boto3_clients, mock_container_runtime, tmp_path
    ):
        """Test graceful handling of ConflictException when agent already exists."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.iam_client = mock_iam_client  # Use provided IAM client
        mock_factory.setup_full_session_mock(mock_boto3_clients)

        # Mock ConflictException on create, then successful list and update
        from botocore.exceptions import ClientError

        conflict_error = ClientError(
            error_response={"Error": {"Code": "ConflictException", "Message": "Agent already exists"}},
            operation_name="CreateAgentRuntime",
        )

        # Mock the bedrock client to throw ConflictException on create_agent_runtime
        mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.side_effect = conflict_error

        # Mock successful list_agent_runtimes to find existing agent
        mock_boto3_clients["bedrock_agentcore"].list_agent_runtimes.return_value = {
            "agentRuntimes": [
                {
                    "agentRuntimeId": "existing-agent-123",
                    "agentRuntimeArn": (
                        "arn:aws:bedrock-agentcore:us-west-2:123456789012:agent-runtime/existing-agent-123"
                    ),
                    "agentRuntimeName": "test-agent",
                }
            ]
        }

        # Mock successful update_agent_runtime
        mock_boto3_clients["bedrock_agentcore"].update_agent_runtime.return_value = {
            "agentRuntimeArn": "arn:aws:bedrock-agentcore:us-west-2:123456789012:agent-runtime/existing-agent-123"
        }

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            result = launch_bedrock_agentcore(
                config_path, local=False, auto_update_on_conflict=True, use_codebuild=False
            )

            # Verify that create was attempted first
            mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.assert_called_once()

            # Verify that list was called to find existing agent
            mock_boto3_clients["bedrock_agentcore"].list_agent_runtimes.assert_called()

            # Verify that update was called instead of failing
            mock_boto3_clients["bedrock_agentcore"].update_agent_runtime.assert_called_once()

            # Verify successful deployment
            assert result.mode == "cloud"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")

        # Verify the config was updated with the discovered agent ID
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        updated_config = load_config(config_path)
        updated_agent = updated_config.agents["test-agent"]
        assert updated_agent.bedrock_agentcore.agent_id == "existing-agent-123"
        assert (
            updated_agent.bedrock_agentcore.agent_arn
            == "arn:aws:bedrock-agentcore:us-west-2:123456789012:agent-runtime/existing-agent-123"
        )

    def test_launch_cloud_conflict_exception_disabled_auto_update(
        self, mock_boto3_clients, mock_container_runtime, tmp_path
    ):
        """Test ConflictException when auto_update_on_conflict is disabled."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.iam_client = mock_iam_client  # Use provided IAM client
        mock_factory.setup_full_session_mock(mock_boto3_clients)

        # Mock ConflictException on create
        from botocore.exceptions import ClientError

        conflict_error = ClientError(
            error_response={"Error": {"Code": "ConflictException", "Message": "Agent already exists"}},
            operation_name="CreateAgentRuntime",
        )
        mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.side_effect = conflict_error

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
        ):
            # Should raise ConflictException when auto_update_on_conflict=False
            with pytest.raises(ClientError, match="ConflictException"):
                launch_bedrock_agentcore(config_path, local=False, auto_update_on_conflict=False)

            # Verify that create was attempted but list/update were not called
            mock_boto3_clients["bedrock_agentcore"].create_agent_runtime.assert_called_once()
            mock_boto3_clients["bedrock_agentcore"].list_agent_runtimes.assert_not_called()
            mock_boto3_clients["bedrock_agentcore"].update_agent_runtime.assert_not_called()

    def test_launch_cloud_with_existing_session_id_reset(self, mock_boto3_clients, mock_container_runtime, tmp_path):
        """Test that session ID gets reset when deploying to cloud."""
        existing_session_id = "existing-session-123"
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
            agent_session_id=existing_session_id,  # Pre-existing session ID
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.iam_client = mock_iam_client  # Use provided IAM client
        mock_factory.setup_full_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.log") as mock_log,
        ):
            result = launch_bedrock_agentcore(config_path, local=False, use_codebuild=False)

            # Verify deployment succeeded
            assert result.mode == "cloud"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")

            # Verify warning log was emitted about session ID reset
            mock_log.warning.assert_called_with(
                "⚠️ Session ID will be reset to connect to the updated agent. "
                "The previous agent remains accessible via the original session ID: %s",
                existing_session_id,
            )

        # Verify the session ID was reset to None in the config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        updated_config = load_config(config_path)
        updated_agent = updated_config.agents["test-agent"]
        assert updated_agent.bedrock_agentcore.agent_session_id is None

    def test_launch_cloud_without_existing_session_id_no_reset(
        self, mock_boto3_clients, mock_container_runtime, tmp_path
    ):
        """Test that no session ID reset occurs when no session ID exists."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
            agent_session_id=None,  # No existing session ID
        )
        create_test_agent_file(tmp_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "Arn": "arn:aws:iam::123456789012:role/TestRole",
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                },
            }
        }

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.iam_client = mock_iam_client  # Use provided IAM client
        mock_factory.setup_full_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.deploy_to_ecr"),
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.log") as mock_log,
        ):
            result = launch_bedrock_agentcore(config_path, local=False, use_codebuild=False)

            # Verify deployment succeeded
            assert result.mode == "cloud"
            assert hasattr(result, "agent_arn")
            assert hasattr(result, "agent_id")

            # Verify NO warning log was emitted about session ID reset
            mock_log.warning.assert_not_called()

        # Verify the session ID remains None in the config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        updated_config = load_config(config_path)
        updated_agent = updated_config.agents["test-agent"]
        assert updated_agent.bedrock_agentcore.agent_session_id is None

    def test_launch_local_mode_no_docker_runtime(self, tmp_path):
        """Test local mode when Docker is not available."""
        config_path = create_test_config(tmp_path)

        # Create a mock runtime without Docker available
        mock_runtime_no_docker = MagicMock()
        mock_runtime_no_docker.runtime = "none"
        mock_runtime_no_docker.has_local_runtime = False  # No Docker available

        with patch(
            "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
            return_value=mock_runtime_no_docker,
        ):
            with pytest.raises(RuntimeError, match="Cannot run locally - no container runtime available"):
                launch_bedrock_agentcore(config_path, local=True)


class TestEnsureExecutionRole:
    """Test _ensure_execution_role functionality."""

    def test_ensure_execution_role_auto_create_success(self, mock_boto3_clients, tmp_path):
        """Test successful execution role auto-creation."""
        config_path = create_test_config(tmp_path, execution_role_auto_create=True)

        # Load the config to get the agent and project configs
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]

        # Role name will use random suffix, so we can't predict the exact name
        created_role_arn = "arn:aws:iam::123456789012:role/AmazonBedrockAgentCoreRuntimeSDKServiceRole-abc123xyz9"

        with (
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
            ) as mock_get_or_create_role,
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.save_config") as mock_save_config,
        ):
            mock_get_or_create_role.return_value = created_role_arn

            result = _ensure_execution_role(
                agent_config=agent_config,
                project_config=project_config,
                config_path=config_path,
                agent_name="test-agent",
                region="us-west-2",
                account_id="123456789012",
            )

            # Verify role creation was called with correct parameters
            call_args = mock_get_or_create_role.call_args
            assert call_args.kwargs["region"] == "us-west-2"
            assert call_args.kwargs["account_id"] == "123456789012"
            assert call_args.kwargs["agent_name"] == "test-agent"
            assert "logger" in call_args.kwargs

            # Verify config was updated
            assert agent_config.aws.execution_role == created_role_arn
            assert agent_config.aws.execution_role_auto_create is False

            # Verify config was saved
            mock_save_config.assert_called_once_with(project_config, config_path)

            # Verify return value
            assert result == created_role_arn

    def test_ensure_execution_role_existing_role_no_create(self, tmp_path):
        """Test when execution role already exists (no auto-creation needed)."""
        existing_role_arn = "arn:aws:iam::123456789012:role/existing-role"

        config_path = create_test_config(
            tmp_path,
            execution_role=existing_role_arn,
            execution_role_auto_create=True,  # Should be ignored
        )

        # Load the config to get the agent and project configs
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]

        # Mock IAM client response for role validation
        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock-agentcore.amazonaws.com"}}]
                }
            }
        }

        with (
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
            ) as mock_create_role,
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.boto3.Session") as mock_session,
        ):
            mock_session.return_value.client.return_value = mock_iam_client

            result = _ensure_execution_role(
                agent_config=agent_config,
                project_config=project_config,
                config_path=config_path,
                agent_name="test-agent",
                region="us-west-2",
                account_id="123456789012",
            )

            # Verify role creation was NOT called
            mock_create_role.assert_not_called()

            # Verify return value is existing role
            assert result == existing_role_arn

    def test_ensure_execution_role_no_role_no_auto_create(self, tmp_path):
        """Test error when no execution role and auto-create disabled."""
        config_path = create_test_config(tmp_path, execution_role_auto_create=False)

        # Load the config to get the agent and project configs
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]

        with pytest.raises(ValueError, match="Execution role not configured and auto-create not enabled"):
            _ensure_execution_role(
                agent_config=agent_config,
                project_config=project_config,
                config_path=config_path,
                agent_name="test-agent",
                region="us-west-2",
                account_id="123456789012",
            )

    def test_ensure_execution_role_creation_failure(self, tmp_path):
        """Test error handling when role creation fails."""
        config_path = create_test_config(tmp_path, execution_role_auto_create=True)

        # Load the config to get the agent and project configs
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]

        with patch(
            "bedrock_agentcore_starter_toolkit.operations.runtime.launch.get_or_create_runtime_execution_role"
        ) as mock_get_or_create_role:
            # Mock role creation failure
            mock_get_or_create_role.side_effect = Exception("IAM permission denied")

            with pytest.raises(Exception, match="IAM permission denied"):
                _ensure_execution_role(
                    agent_config=agent_config,
                    project_config=project_config,
                    config_path=config_path,
                    agent_name="test-agent",
                    region="us-west-2",
                    account_id="123456789012",
                )

    def test_validate_execution_role_url_encoded_policy(self):
        """Test _validate_execution_role with URL-encoded trust policy."""
        import json
        import urllib.parse

        from bedrock_agentcore_starter_toolkit.operations.runtime.launch import _validate_execution_role

        # Create URL-encoded trust policy
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        url_encoded_policy = urllib.parse.quote(json.dumps(trust_policy))

        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": url_encoded_policy  # URL-encoded string
            }
        }

        mock_session = MagicMock()
        mock_session.client.return_value = mock_iam_client

        result = _validate_execution_role("arn:aws:iam::123456789012:role/test-role", mock_session)

        assert result is True

    def test_validate_execution_role_invalid_trust_policy(self):
        """Test _validate_execution_role with invalid trust policy."""
        from bedrock_agentcore_starter_toolkit.operations.runtime.launch import _validate_execution_role

        mock_iam_client = MagicMock()
        mock_iam_client.get_role.return_value = {
            "Role": {
                "AssumeRolePolicyDocument": {
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"Service": "lambda.amazonaws.com"},  # Wrong service
                            "Action": "sts:AssumeRole",
                        }
                    ]
                }
            }
        }

        mock_session = MagicMock()
        mock_session.client.return_value = mock_iam_client

        result = _validate_execution_role("arn:aws:iam::123456789012:role/test-role", mock_session)

        assert result is False

    def test_validate_execution_role_role_not_found(self):
        """Test _validate_execution_role when role doesn't exist."""
        from botocore.exceptions import ClientError

        from bedrock_agentcore_starter_toolkit.operations.runtime.launch import _validate_execution_role

        mock_iam_client = MagicMock()
        mock_iam_client.get_role.side_effect = ClientError({"Error": {"Code": "NoSuchEntity"}}, "GetRole")

        mock_session = MagicMock()
        mock_session.client.return_value = mock_iam_client

        result = _validate_execution_role("arn:aws:iam::123456789012:role/nonexistent-role", mock_session)

        assert result is False


class TestXRayService:
    """Test X-Ray service functionality."""

    def test_validate_aws_credentials_success(self):
        """Test successful AWS credentials validation."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_session.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

        result = validate_aws_credentials(mock_session)

        assert result is True
        mock_session.client.assert_called_once_with("sts")
        mock_sts_client.get_caller_identity.assert_called_once()

    def test_validate_aws_credentials_no_credentials(self):
        """Test credentials validation when no credentials available."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_session.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.side_effect = NoCredentialsError()

        result = validate_aws_credentials(mock_session)

        assert result is False

    def test_validate_aws_credentials_client_error(self):
        """Test credentials validation with ClientError."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_session.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied"}}, operation_name="GetCallerIdentity"
        )

        result = validate_aws_credentials(mock_session)

        assert result is False

    def test_validate_aws_credentials_botoccore_error(self):
        """Test credentials validation with BotoCoreError."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_session.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.side_effect = BotoCoreError()

        result = validate_aws_credentials(mock_session)

        assert result is False

    def test_is_xray_configured_true(self):
        """Test X-Ray configuration check when already configured."""
        mock_session = MagicMock()
        mock_xray_client = MagicMock()
        mock_session.client.return_value = mock_xray_client
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "CloudWatchLogs"}

        result = is_xray_transaction_search_configured(mock_session)

        assert result is True
        mock_session.client.assert_called_once_with("xray")
        mock_xray_client.get_trace_segment_destination.assert_called_once()

    def test_is_xray_configured_false(self):
        """Test X-Ray configuration check when not configured."""
        mock_session = MagicMock()
        mock_xray_client = MagicMock()
        mock_session.client.return_value = mock_xray_client
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}

        result = is_xray_transaction_search_configured(mock_session)

        assert result is False

    def test_is_xray_configured_client_error(self):
        """Test X-Ray configuration check with ClientError."""
        mock_session = MagicMock()
        mock_xray_client = MagicMock()
        mock_session.client.return_value = mock_xray_client
        mock_xray_client.get_trace_segment_destination.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied"}}, operation_name="GetTraceSegmentDestination"
        )

        result = is_xray_transaction_search_configured(mock_session)

        assert result is False

    def test_setup_xray_success(self):
        """Test successful X-Ray setup."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}  # Not configured

        success, message = setup_xray_transaction_search(mock_session)

        assert success is True
        assert message == "X-Ray Transaction Search configured successfully"
        mock_xray_client.update_trace_segment_destination.assert_called_once_with(Destination="CloudWatchLogs")
        mock_xray_client.update_indexing_rule.assert_called_once_with(
            Name="Default", Rule={"Probabilistic": {"DesiredSamplingPercentage": 1}}
        )

    def test_setup_xray_already_configured(self):
        """Test X-Ray setup when already configured."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "CloudWatchLogs"}

        success, message = setup_xray_transaction_search(mock_session)

        assert success is True
        assert message == "X-Ray Transaction Search already configured"
        mock_xray_client.update_trace_segment_destination.assert_not_called()
        mock_xray_client.update_indexing_rule.assert_not_called()

    def test_setup_xray_credentials_unavailable(self):
        """Test X-Ray setup when credentials unavailable."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_session.client.return_value = mock_sts_client
        mock_sts_client.get_caller_identity.side_effect = NoCredentialsError()

        success, message = setup_xray_transaction_search(mock_session)

        assert success is False
        assert "AWS credentials not available" in message

    def test_setup_xray_trace_destination_access_denied(self):
        """Test X-Ray setup with access denied on trace destination."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}
        mock_xray_client.update_trace_segment_destination.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            operation_name="UpdateTraceSegmentDestination",
        )

        success, message = setup_xray_transaction_search(mock_session)

        assert success is False
        assert "Insufficient permissions for X-Ray setup: Access denied" in message

    def test_setup_xray_indexing_rule_validation_error(self):
        """Test X-Ray setup with validation error on indexing rule."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}
        mock_xray_client.update_indexing_rule.side_effect = ClientError(
            error_response={"Error": {"Code": "ValidationException", "Message": "Invalid rule"}},
            operation_name="UpdateIndexingRule",
        )

        success, message = setup_xray_transaction_search(mock_session)

        assert success is False
        assert "Invalid indexing rule configuration: Invalid rule" in message

    def test_setup_xray_botoccore_error(self):
        """Test X-Ray setup with BotoCoreError."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}
        mock_xray_client.update_trace_segment_destination.side_effect = BotoCoreError()

        success, message = setup_xray_transaction_search(mock_session)

        assert success is False
        assert "AWS service error configuring X-Ray Transaction Search" in message

    def test_setup_xray_unexpected_error(self):
        """Test X-Ray setup with unexpected error."""
        mock_session = MagicMock()
        mock_sts_client = MagicMock()
        mock_xray_client = MagicMock()

        def mock_client(service):
            if service == "sts":
                return mock_sts_client
            elif service == "xray":
                return mock_xray_client

        mock_session.client.side_effect = mock_client
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "S3"}
        mock_xray_client.update_trace_segment_destination.side_effect = Exception("Unexpected error")

        success, message = setup_xray_transaction_search(mock_session)

        assert success is False
        assert "Unexpected error configuring X-Ray Transaction Search: Unexpected error" in message

    def test_setup_xray_default_session(self):
        """Test X-Ray setup with default session creation."""
        with patch("bedrock_agentcore_starter_toolkit.services.xray.boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session_cls.return_value = mock_session
            mock_sts_client = MagicMock()
            mock_xray_client = MagicMock()

            def mock_client(service):
                if service == "sts":
                    return mock_sts_client
                elif service == "xray":
                    return mock_xray_client

            mock_session.client.side_effect = mock_client
            mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
            mock_xray_client.get_trace_segment_destination.return_value = {"Destination": "CloudWatchLogs"}

            success, message = setup_xray_transaction_search()  # No session provided

            assert success is True
            assert message == "X-Ray Transaction Search already configured"
            # Session is created once in setup_xray_transaction_search when no session is provided
            assert mock_session_cls.call_count >= 1


class TestLaunchWithObservability:
    """Test launch integration with observability features."""

    def test_launch_cloud_with_observability_setup_success(self, mock_boto3_clients, tmp_path):
        """Test cloud deployment with successful observability setup."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Enable observability in config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config, save_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]
        agent_config.aws.observability.enabled = True
        project_config.agents["test-agent"] = agent_config
        save_config(project_config, config_path)

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository") as mock_create_ecr,
            patch("bedrock_agentcore_starter_toolkit.services.xray.setup_xray_transaction_search") as mock_setup_xray,
        ):
            mock_create_ecr.return_value = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo"
            mock_setup_xray.return_value = (True, "X-Ray configured successfully")

            result = launch_bedrock_agentcore(config_path, local=False)

            # Verify observability setup was called
            mock_setup_xray.assert_called_once()
            call_args = mock_setup_xray.call_args[0]
            assert len(call_args) == 1  # session parameter
            session = call_args[0]
            assert session.region_name == "us-west-2"

            # Verify deployment succeeded
            assert result.mode == "codebuild"
            assert hasattr(result, "agent_arn")

    def test_launch_cloud_with_observability_setup_failure(self, mock_boto3_clients, tmp_path):
        """Test cloud deployment continues when observability setup fails."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Enable observability in config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config, save_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]
        agent_config.aws.observability.enabled = True
        project_config.agents["test-agent"] = agent_config
        save_config(project_config, config_path)

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository") as mock_create_ecr,
            patch("bedrock_agentcore_starter_toolkit.services.xray.setup_xray_transaction_search") as mock_setup_xray,
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.log") as mock_log,
        ):
            mock_create_ecr.return_value = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo"
            mock_setup_xray.return_value = (False, "Access denied for X-Ray setup")

            result = launch_bedrock_agentcore(config_path, local=False)

            # Verify observability setup was called and failed
            mock_setup_xray.assert_called_once()

            # Verify warning was logged but deployment continued
            mock_log.warning.assert_called_with("⚠️ Observability setup failed: %s", "Access denied for X-Ray setup")

            # Verify deployment succeeded despite observability failure
            assert result.mode == "codebuild"
            assert hasattr(result, "agent_arn")

    def test_launch_cloud_observability_disabled(self, mock_boto3_clients, tmp_path):
        """Test cloud deployment with observability disabled."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Observability is disabled by default in create_test_config

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository") as mock_create_ecr,
            patch("bedrock_agentcore_starter_toolkit.services.xray.setup_xray_transaction_search") as mock_setup_xray,
        ):
            mock_create_ecr.return_value = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo"

            result = launch_bedrock_agentcore(config_path, local=False)

            # Verify observability setup was NOT called
            mock_setup_xray.assert_not_called()

            # Verify deployment succeeded
            assert result.mode == "codebuild"
            assert hasattr(result, "agent_arn")

    def test_launch_local_no_observability_setup(self, mock_container_runtime, tmp_path):
        """Test local deployment skips observability setup."""
        config_path = create_test_config(tmp_path)
        create_test_agent_file(tmp_path)

        # Enable observability in config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config, save_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]
        agent_config.aws.observability.enabled = True
        project_config.agents["test-agent"] = agent_config
        save_config(project_config, config_path)

        # Mock the build to return success
        mock_container_runtime.build.return_value = (True, ["Successfully built test-image"])

        with (
            patch(
                "bedrock_agentcore_starter_toolkit.operations.runtime.launch.ContainerRuntime",
                return_value=mock_container_runtime,
            ),
            patch("bedrock_agentcore_starter_toolkit.services.xray.setup_xray_transaction_search") as mock_setup_xray,
        ):
            result = launch_bedrock_agentcore(config_path, local=True)

            # Verify observability setup was NOT called for local mode
            mock_setup_xray.assert_not_called()

            # Verify local deployment succeeded
            assert result.mode == "local"
            assert result.tag == "bedrock_agentcore-test-agent:latest"

    def test_launch_cloud_with_observability_boto3_session_reuse(self, mock_boto3_clients, tmp_path):
        """Test that observability setup reuses the same boto3 session pattern as other AWS operations."""
        config_path = create_test_config(
            tmp_path,
            execution_role="arn:aws:iam::123456789012:role/TestRole",
            ecr_repository="123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo",
        )
        create_test_agent_file(tmp_path)

        # Enable observability in config
        from bedrock_agentcore_starter_toolkit.utils.runtime.config import load_config, save_config

        project_config = load_config(config_path)
        agent_config = project_config.agents["test-agent"]
        agent_config.aws.observability.enabled = True
        project_config.agents["test-agent"] = agent_config
        save_config(project_config, config_path)

        # Setup mock AWS clients
        mock_factory = MockAWSClientFactory()
        mock_factory.setup_session_mock(mock_boto3_clients)

        with (
            patch("bedrock_agentcore_starter_toolkit.services.ecr.get_or_create_ecr_repository") as mock_create_ecr,
            patch("bedrock_agentcore_starter_toolkit.services.xray.setup_xray_transaction_search") as mock_setup_xray,
            patch("bedrock_agentcore_starter_toolkit.operations.runtime.launch.boto3.Session") as mock_session_cls,
        ):
            mock_create_ecr.return_value = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test-repo"
            mock_setup_xray.return_value = (True, "X-Ray configured successfully")

            # Create a mock session that will be returned by boto3.Session()
            mock_session_instance = MagicMock()
            mock_session_instance.region_name = "us-west-2"
            mock_session_cls.return_value = mock_session_instance

            launch_bedrock_agentcore(config_path, local=False)

            # Verify that observability setup was called with a session
            mock_setup_xray.assert_called_once()
            call_args = mock_setup_xray.call_args[0]
            assert len(call_args) == 1
            session = call_args[0]
            assert session.region_name == "us-west-2"
