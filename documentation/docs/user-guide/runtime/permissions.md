# Runtime Permissions

This guide covers the IAM permissions required to run agents with Amazon Bedrock AgentCore Runtime. The toolkit supports two types of execution roles with distinct permission sets.

!!! info "AWS Documentation Reference"
    This guide is based on the official [AWS Bedrock AgentCore Runtime Permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html) documentation, with additional details specific to the toolkit's implementation.

## Overview

The toolkit requires two types of IAM roles for different phases of agent deployment:

- **Runtime Execution Role**: Used by Bedrock AgentCore Runtime to execute your agent
- **CodeBuild Execution Role**: Used by AWS CodeBuild to build and push container images (ARM64 architecture)

Both roles can be automatically created by the toolkit or manually specified using existing roles.

## Runtime Execution Role

The Runtime Execution Role is assumed by the Bedrock AgentCore service to run your agent. This role requires specific permissions to access AWS services on behalf of your agent.

### Trust Policy

The runtime execution role must trust the `bedrock-agentcore.amazonaws.com` service:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AssumeRolePolicy",
      "Effect": "Allow",
      "Principal": {
        "Service": "bedrock-agentcore.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "aws:SourceAccount": "YOUR_ACCOUNT_ID"
        },
        "ArnLike": {
          "aws:SourceArn": "arn:aws:bedrock-agentcore:YOUR_REGION:YOUR_ACCOUNT_ID:*"
        }
      }
    }
  ]
}
```

### Permissions Policy

The runtime execution role requires the following permissions:

#### ECR Container Access
```json
{
  "Sid": "ECRImageAccess",
  "Effect": "Allow",
  "Action": [
    "ecr:BatchGetImage",
    "ecr:GetDownloadUrlForLayer"
  ],
  "Resource": [
    "arn:aws:ecr:YOUR_REGION:YOUR_ACCOUNT_ID:repository/*"
  ]
},
{
  "Sid": "ECRTokenAccess",
  "Effect": "Allow",
  "Action": [
    "ecr:GetAuthorizationToken"
  ],
  "Resource": "*"
}
```

**Purpose**: Allows the runtime to pull your agent's container image from Amazon ECR.

#### CloudWatch Logs
```json
{
  "Effect": "Allow",
  "Action": [
    "logs:DescribeLogStreams",
    "logs:CreateLogGroup"
  ],
  "Resource": [
    "arn:aws:logs:YOUR_REGION:YOUR_ACCOUNT_ID:log-group:/aws/bedrock-agentcore/runtimes/*"
  ]
},
{
  "Effect": "Allow",
  "Action": [
    "logs:DescribeLogGroups"
  ],
  "Resource": [
    "arn:aws:logs:YOUR_REGION:YOUR_ACCOUNT_ID:log-group:*"
  ]
},
{
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": [
    "arn:aws:logs:YOUR_REGION:YOUR_ACCOUNT_ID:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*"
  ]
}
```

**Purpose**: Enables your agent to write logs to CloudWatch for monitoring and debugging.

#### Observability and Monitoring
```json
{
  "Effect": "Allow",
  "Action": [
    "xray:PutTraceSegments",
    "xray:PutTelemetryRecords",
    "xray:GetSamplingRules",
    "xray:GetSamplingTargets"
  ],
  "Resource": ["*"]
},
{
  "Effect": "Allow",
  "Resource": "*",
  "Action": "cloudwatch:PutMetricData",
  "Condition": {
    "StringEquals": {
      "cloudwatch:namespace": "bedrock-agentcore"
    }
  }
}
```

**Purpose**: Enables distributed tracing with X-Ray and custom metrics reporting to CloudWatch.

#### Workload Identity
```json
{
  "Sid": "GetAgentAccessToken",
  "Effect": "Allow",
  "Action": [
    "bedrock-agentcore:GetWorkloadAccessToken",
    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
    "bedrock-agentcore:GetWorkloadAccessTokenForUserId"
  ],
  "Resource": [
    "arn:aws:bedrock-agentcore:YOUR_REGION:YOUR_ACCOUNT_ID:workload-identity-directory/default",
    "arn:aws:bedrock-agentcore:YOUR_REGION:YOUR_ACCOUNT_ID:workload-identity-directory/default/workload-identity/YOUR_AGENT_NAME-*"
  ]
}
```

**Purpose**: Allows your agent to obtain identity tokens for secure service-to-service communication.

#### Bedrock Model Access
```json
{
  "Sid": "BedrockModelInvocation",
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:InvokeModelWithResponseStream"
  ],
  "Resource": [
    "arn:aws:bedrock:*::foundation-model/*",
    "arn:aws:bedrock:YOUR_REGION:YOUR_ACCOUNT_ID:*"
  ]
}
```

**Purpose**: Enables your agent to invoke foundation models and custom models in Amazon Bedrock.

## CodeBuild Execution Role

The CodeBuild Execution Role is used by AWS CodeBuild to build your agent's Docker container for ARM64 architecture and push it to Amazon ECR.

### Trust Policy

The CodeBuild execution role must trust the `codebuild.amazonaws.com` service:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "codebuild.amazonaws.com"
      },
      "Action": "sts:AssumeRole",
      "Condition": {
        "StringEquals": {
          "aws:SourceAccount": "YOUR_ACCOUNT_ID"
        }
      }
    }
  ]
}
```

### Permissions Policy

The CodeBuild execution role requires the following permissions:

#### ECR Repository Access
```json
{
  "Effect": "Allow",
  "Action": [
    "ecr:GetAuthorizationToken"
  ],
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
  "Resource": "arn:aws:ecr:YOUR_REGION:YOUR_ACCOUNT_ID:repository/YOUR_ECR_REPOSITORY"
}
```

**Purpose**: Allows CodeBuild to authenticate with ECR and push the built container image.

#### CloudWatch Logs for Build Process
```json
{
  "Effect": "Allow",
  "Action": [
    "logs:CreateLogGroup",
    "logs:CreateLogStream",
    "logs:PutLogEvents"
  ],
  "Resource": [
    "arn:aws:logs:YOUR_REGION:YOUR_ACCOUNT_ID:log-group:/aws/codebuild/bedrock-agentcore-*"
  ]
}
```

**Purpose**: Enables CodeBuild to create and write to log groups for build monitoring.

#### S3 Source Access
```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject"
  ],
  "Resource": [
    "arn:aws:s3:::bedrock-agentcore-codebuild-sources-YOUR_ACCOUNT_ID-YOUR_REGION/*"
  ]
}
```

**Purpose**: Allows CodeBuild to access the source code uploaded to the toolkit's managed S3 bucket.

## Toolkit Implementation Details

### Role Naming Convention

The toolkit uses deterministic naming for auto-created roles:

- **Runtime Role**: `AmazonBedrockAgentCoreSDKRuntime-{region}-{hash}`
- **CodeBuild Role**: `AmazonBedrockAgentCoreSDKCodeBuild-{region}-{hash}`

Where `{hash}` is a deterministic 10-character hash based on your agent name, ensuring consistent role names across deployments.

### Auto-Creation vs Manual Roles

#### Auto-Created Roles
- Use the exact policies shown above
- Created with appropriate resource scoping
- Include all required permissions for toolkit functionality

#### Manual Roles
- Must include minimum required permissions from the policies above
- Trust policies must allow the appropriate AWS services
- Resource ARNs must be updated to match your specific resources

## Configuration Examples

### Using Auto-Created Roles
```yaml
# .bedrock_agentcore.yaml
agents:
  my-agent:
    aws:
      execution_role_auto_create: true
      region: us-west-2
      account: "123456789012"
```

### Using Existing Roles
```yaml
# .bedrock_agentcore.yaml
agents:
  my-agent:
    aws:
      execution_role: "arn:aws:iam::123456789012:role/MyCustomRole"
      region: us-west-2
      account: "123456789012"
```

### CLI Configuration
```bash
# Auto-create roles
bedrock-agentcore configure --entrypoint agent.py

# Use existing role
bedrock-agentcore configure --entrypoint agent.py --execution-role MyCustomRole
```
