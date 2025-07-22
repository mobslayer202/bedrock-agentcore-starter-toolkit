# Changelog

## [0.1.1] - 2025-01-22

### Added
- **Multi-platform Docker build support via AWS CodeBuild** (#1)
  - New `--codebuild` flag for `agentcore launch` command enables ARM64 container builds
  - Complete `CodeBuildService` class with ARM64-optimized build pipeline
  - Automated infrastructure provisioning (S3 buckets, IAM roles, CodeBuild projects)
  - ARM64-optimized buildspec with Docker BuildKit caching and parallel push operations
  - Smart source management with .dockerignore pattern support and S3 lifecycle policies
  - Real-time build monitoring with detailed phase tracking
  - Support for `aws/codebuild/amazonlinux2-aarch64-standard:3.0` image
  - ECR caching strategy for faster ARM64 builds

- **Automatic IAM execution role creation** (#2)
  - Auto-creation of IAM execution roles for Bedrock AgentCore Runtime
  - Policy templates for execution role and trust policy
  - Detailed logging and progress tracking during role creation
  - Informative error messages for common IAM scenarios
  - Eliminates need for manual IAM role creation before deployment

### Changed
- Enhanced `agentcore launch` command to support both local Docker and CodeBuild workflows
- Improved error handling patterns throughout the codebase
- Updated AWS SDK exception handling to use standard `ClientError` patterns instead of service-specific exceptions

### Fixed
- Fixed AWS IAM exception handling by replacing problematic service-specific exceptions with standard `ClientError` patterns
- Resolved pre-commit hook compliance issues with proper code formatting

### Improved
- Added 90%+ test coverage with 20+ new comprehensive test cases
- Enhanced error handling with proper AWS SDK patterns
- Improved build reliability and monitoring capabilities
- Better user experience with one-command ARM64 deployment

## [0.1.0] - 2025-01-16

### Added
- Initial release of Bedrock AgentCore Starter Toolkit
- CLI toolkit for deploying AI agents to Amazon Bedrock AgentCore
- Zero infrastructure management with built-in gateway and memory integrations
- Support for popular frameworks (Strands, LangGraph, CrewAI, custom agents)
- Core CLI commands: `configure`, `launch`, `invoke`, `status`
- Local testing capabilities with `--local` flag
- Integration with Bedrock AgentCore SDK
- Basic Docker containerization support
- Comprehensive documentation and examples
