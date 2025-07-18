"""CodeBuild service for ARM64 container builds."""

import json
import logging
import os
import tempfile
import time
import zipfile
import fnmatch
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import boto3
from botocore.exceptions import ClientError


class CodeBuildService:
    """Service for managing CodeBuild projects and builds for ARM64."""
    
    def __init__(self, session: boto3.Session):
        self.session = session
        self.client = session.client('codebuild')
        self.s3_client = session.client('s3')
        self.iam_client = session.client('iam')
        self.logger = logging.getLogger(__name__)
        self.source_bucket = None
    
    def get_source_bucket_name(self, account_id: str) -> str:
        """Get S3 bucket name for CodeBuild sources."""
        region = self.session.region_name
        return f"bedrock-agentcore-codebuild-sources-{account_id}-{region}"
    
    def ensure_source_bucket(self, account_id: str) -> str:
        """Ensure S3 bucket exists for CodeBuild sources."""
        bucket_name = self.get_source_bucket_name(account_id)
        
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            self.logger.debug(f"Using existing S3 bucket: {bucket_name}")
        except ClientError:
            # Create bucket
            region = self.session.region_name
            if region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': region}
                )
            
            # Set lifecycle to cleanup old builds
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=bucket_name,
                LifecycleConfiguration={
                    'Rules': [{
                        'ID': 'DeleteOldBuilds',
                        'Status': 'Enabled',
                        'Filter': {},
                        'Expiration': {'Days': 7}
                    }]
                }
            )
            
            self.logger.info(f"Created S3 bucket: {bucket_name}")
        
        return bucket_name
    
    def upload_source(self, agent_name: str) -> str:
        """Upload current directory to S3, respecting .dockerignore patterns."""
        account_id = self.session.client('sts').get_caller_identity()['Account']
        bucket_name = self.ensure_source_bucket(account_id)
        self.source_bucket = bucket_name
        
        # Parse .dockerignore patterns
        ignore_patterns = self._parse_dockerignore()
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
            try:
                with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk('.'):
                        # Convert to relative path
                        rel_root = os.path.relpath(root, '.')
                        if rel_root == '.':
                            rel_root = ''
                        
                        # Filter directories
                        dirs[:] = [d for d in dirs if not self._should_ignore(
                            os.path.join(rel_root, d) if rel_root else d, 
                            ignore_patterns, 
                            is_dir=True
                        )]
                        
                        for file in files:
                            file_rel_path = os.path.join(rel_root, file) if rel_root else file
                            
                            # Skip if matches ignore pattern
                            if self._should_ignore(file_rel_path, ignore_patterns, is_dir=False):
                                continue
                            
                            file_path = Path(root) / file
                            zipf.write(file_path, file_rel_path)
                
                # Create agent-organized S3 key: agentname/timestamp.zip
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
                s3_key = f"{agent_name}/{timestamp}.zip"
                
                self.s3_client.upload_file(temp_zip.name, bucket_name, s3_key)
                
                self.logger.info(f"Uploaded source to S3: {s3_key}")
                return f"s3://{bucket_name}/{s3_key}"
                
            finally:
                os.unlink(temp_zip.name)
    
    def _normalize_s3_location(self, source_location: str) -> str:
        """Convert s3:// URL to bucket/key format for CodeBuild."""
        return source_location.replace('s3://', '') if source_location.startswith('s3://') else source_location

    def create_codebuild_execution_role(self, account_id: str, ecr_repository_arn: str) -> str:
        """Auto-create CodeBuild execution role."""
        region = self.session.region_name
        role_name = f"BedrockAgentCoreCodeBuildExecutionRole-{region}"
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "codebuild.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account_id
                    }
                }
            }]
        }
        
        permissions_policy = {
            "Version": "2012-10-17", 
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["ecr:GetAuthorizationToken"],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:PutImage",
                        "ecr:InitiateLayerUpload",
                        "ecr:UploadLayerPart",
                        "ecr:CompleteLayerUpload"
                    ],
                    "Resource": ecr_repository_arn
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": f"arn:aws:logs:{region}:{account_id}:log-group:/aws/codebuild/bedrock-agentcore-*"
                },
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject"],
                    "Resource": f"arn:aws:s3:::{self.get_source_bucket_name(account_id)}/*"
                }
            ]
        }
        
        try:
            # Create role
            self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="CodeBuild execution role for Bedrock AgentCore ARM64 builds"
            )
            
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName="CodeBuildExecutionPolicy", 
                PolicyDocument=json.dumps(permissions_policy)
            )
            
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            self.logger.info(f"Created CodeBuild execution role: {role_arn}")
            
            # Wait for IAM propagation to prevent CodeBuild authorization errors
            self.logger.info("Waiting for IAM role propagation...")
            time.sleep(15)
            
            return role_arn
            
        except self.iam_client.exceptions.EntityAlreadyExistsException:
            # Role exists, return ARN
            role_arn = f"arn:aws:iam::{account_id}:role/{role_name}"
            self.logger.info(f"Using existing CodeBuild execution role: {role_arn}")
            return role_arn
    
    def create_or_update_project(
        self, 
        agent_name: str, 
        ecr_repository_uri: str, 
        execution_role: str,
        source_location: str
    ) -> str:
        """Create or update CodeBuild project for ARM64 builds."""
        
        project_name = f"bedrock-agentcore-{agent_name}-builder"
        
        buildspec = self._get_arm64_buildspec(ecr_repository_uri)
        
        # CodeBuild expects S3 location without s3:// prefix (bucket/key format)
        codebuild_source_location = self._normalize_s3_location(source_location)
        
        project_config = {
            'name': project_name,
            'source': {
                'type': 'S3',
                'location': codebuild_source_location,
                'buildspec': buildspec,
            },
            'artifacts': {
                'type': 'NO_ARTIFACTS',
            },
            'environment': {
                'type': 'ARM_CONTAINER',  # ARM64 images require ARM_CONTAINER environment type
                'image': 'aws/codebuild/amazonlinux2-aarch64-standard:3.0',
                'computeType': 'BUILD_GENERAL1_LARGE',  # 4 vCPUs, 7GB RAM for faster builds
                'privilegedMode': True,  # Required for Docker
            },
            'serviceRole': execution_role,
        }
        
        try:
            self.client.create_project(**project_config)
            self.logger.info(f"Created CodeBuild project: {project_name}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceAlreadyExistsException':
                self.client.update_project(**project_config)
                self.logger.info(f"Updated CodeBuild project: {project_name}")
            else:
                raise
        
        return project_name
    
    def start_build(self, project_name: str, source_location: str) -> str:
        """Start a CodeBuild build."""
        
        # CodeBuild expects S3 location without s3:// prefix (bucket/key format)
        codebuild_source_location = self._normalize_s3_location(source_location)
        
        response = self.client.start_build(
            projectName=project_name,
            sourceLocationOverride=codebuild_source_location,
        )
        
        return response['build']['id']
    
    def wait_for_completion(self, build_id: str, timeout: int = 900):
        """Wait for CodeBuild to complete with detailed phase tracking."""
        
        self.logger.info("Starting CodeBuild monitoring...")
        
        # Phase tracking variables
        current_phase = None
        phase_start_time = None
        build_start_time = time.time()
        
        while time.time() - build_start_time < timeout:
            response = self.client.batch_get_builds(ids=[build_id])
            build = response['builds'][0]
            status = build['buildStatus'] 
            build_phase = build.get('currentPhase', 'UNKNOWN')
            
            # Track phase changes
            if build_phase != current_phase:
                # Log previous phase completion (if any)
                if current_phase and phase_start_time:
                    phase_duration = time.time() - phase_start_time
                    self.logger.info(f"✅ {current_phase} completed in {phase_duration:.1f}s")
                
                # Log new phase start
                current_phase = build_phase
                phase_start_time = time.time()
                total_duration = phase_start_time - build_start_time
                self.logger.info(f"🔄 {current_phase} started (total: {total_duration:.0f}s)")
            
            # Check for completion
            if status == 'SUCCEEDED':
                # Log final phase completion
                if current_phase and phase_start_time:
                    phase_duration = time.time() - phase_start_time
                    self.logger.info(f"✅ {current_phase} completed in {phase_duration:.1f}s")
                
                total_duration = time.time() - build_start_time
                minutes, seconds = divmod(int(total_duration), 60)
                self.logger.info(f"🎉 CodeBuild completed successfully in {minutes}m {seconds}s")
                return
                
            elif status in ['FAILED', 'FAULT', 'STOPPED', 'TIMED_OUT']:
                # Log failure with phase info
                if current_phase:
                    self.logger.error(f"❌ Build failed during {current_phase} phase")
                raise RuntimeError(f"CodeBuild failed with status: {status}")
            
            time.sleep(5)
        
        total_duration = time.time() - build_start_time
        minutes, seconds = divmod(int(total_duration), 60) 
        raise TimeoutError(f"CodeBuild timed out after {minutes}m {seconds}s (current phase: {current_phase})")
    
    def _get_arm64_buildspec(self, ecr_repository_uri: str) -> str:
        """Get optimized buildspec for ARM64 Docker."""
        return f"""
version: 0.2
phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin {ecr_repository_uri}
      - echo Pulling cache image...
      - export DOCKER_BUILDKIT=1
      - export BUILDKIT_PROGRESS=plain
      - docker pull {ecr_repository_uri}:cache || true
  build:
    commands:
      - echo Build started on `date`
      - echo Building ARM64 Docker image with BuildKit processing...
      - export DOCKER_BUILDKIT=1
      - docker buildx create --name arm64builder --use || true
      - docker buildx build --platform linux/arm64 --cache-from {ecr_repository_uri}:cache --load -t bedrock-agentcore-arm64 .
      - docker tag bedrock-agentcore-arm64:latest {ecr_repository_uri}:latest
      - docker tag bedrock-agentcore-arm64:latest {ecr_repository_uri}:cache
  post_build:
    commands:
      - echo Build completed on `date`
      - echo Pushing ARM64 image to ECR...
      - docker push {ecr_repository_uri}:latest &
      - docker push {ecr_repository_uri}:cache &
      - wait
"""
    
    def _parse_dockerignore(self) -> List[str]:
        """Parse .dockerignore file and return list of patterns."""
        dockerignore_path = Path('.dockerignore')
        patterns = []
        
        if dockerignore_path.exists():
            with open(dockerignore_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
            
            self.logger.info(f"Using .dockerignore with {len(patterns)} patterns")
        else:
            # Default patterns if no .dockerignore
            patterns = [
                '.git',
                '__pycache__',
                '*.pyc',
                '.DS_Store',
                'node_modules',
                '.venv',
                'venv',
                '*.egg-info',
                '.bedrock_agentcore.yaml'  # Always exclude config
            ]
            self.logger.info("No .dockerignore found, using default exclude patterns")
        
        return patterns
    
    def _should_ignore(self, path: str, patterns: List[str], is_dir: bool = False) -> bool:
        """Check if path should be ignored based on dockerignore patterns."""
        
        # Normalize path
        if path.startswith('./'):
            path = path[2:]
        
        for pattern in patterns:
            # Handle negation patterns
            if pattern.startswith('!'):
                if self._matches_pattern(path, pattern[1:], is_dir):
                    return False
                continue
            
            # Regular ignore patterns
            if self._matches_pattern(path, pattern, is_dir):
                return True
        
        return False
    
    def _matches_pattern(self, path: str, pattern: str, is_dir: bool) -> bool:
        """Check if path matches a dockerignore pattern."""
        
        # Directory-specific patterns
        if pattern.endswith('/'):
            if not is_dir:
                return False
            pattern = pattern[:-1]
        
        # Exact match
        if path == pattern:
            return True
        
        # Glob pattern match
        if fnmatch.fnmatch(path, pattern):
            return True
        
        # Directory prefix match
        if is_dir and pattern in path.split('/'):
            return True
        
        # File in ignored directory
        if not is_dir and any(fnmatch.fnmatch(part, pattern) for part in path.split('/')):
            return True
        
        return False
