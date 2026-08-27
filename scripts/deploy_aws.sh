#!/usr/bin/env bash
set -euo pipefail

aws_profile="${1:-default}"
aws_region="${2:-us-east-1}"

sam validate --lint --template-file infra/template.yaml
sam build --template-file infra/template.yaml
sam deploy \
  --stack-name aegisland-evidence \
  --profile "$aws_profile" \
  --region "$aws_region" \
  --resolve-s3 \
  --capabilities CAPABILITY_IAM \
  --confirm-changeset

