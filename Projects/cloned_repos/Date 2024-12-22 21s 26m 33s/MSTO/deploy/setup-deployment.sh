#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
CLUSTER_NAME="$APP_NAME-cluster"
SERVICE_NAME="$APP_NAME-service"
TASK_FAMILY="$APP_NAME-task"
CONTAINER_PORT=8080
HOST_PORT=80
MIN_CAPACITY=1
MAX_CAPACITY=4
DESIRED_COUNT=1
CPU_UNITS=256
MEMORY_MB=512

echo "Setting up deployment configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Create ECR repository
echo "Creating ECR repository..."
aws ecr create-repository \
    --repository-name $APP_NAME \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256 \
    --region $REGION

# Create task definition
echo "Creating task definition..."
cat > deploy/task-definition.json << EOF
{
    "family": "$TASK_FAMILY",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "$CPU_UNITS",
    "memory": "$MEMORY_MB",
    "executionRoleArn": "$TASK_EXECUTION_ROLE_ARN",
    "taskRoleArn": "$TASK_ROLE_ARN",
    "containerDefinitions": [
        {
            "name": "$APP_NAME",
            "image": "$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:latest",
            "essential": true,
            "portMappings": [
                {
                    "containerPort": $CONTAINER_PORT,
                    "hostPort": $CONTAINER_PORT,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "AWS_REGION",
                    "value": "$REGION"
                }
            ],
            "mountPoints": [
                {
                    "sourceVolume": "data",
                    "containerPath": "/app/data",
                    "readOnly": false
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/$APP_NAME",
                    "awslogs-region": "$REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:$CONTAINER_PORT/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3,
                "startPeriod": 60
            }
        }
    ],
    "volumes": [
        {
            "name": "data",
            "efsVolumeConfiguration": {
                "fileSystemId": "$EFS_FILESYSTEM_ID",
                "rootDirectory": "/",
                "transitEncryption": "ENABLED"
            }
        }
    ]
}
EOF

# Register task definition
TASK_DEFINITION_ARN=$(aws ecs register-task-definition \
    --cli-input-json file://deploy/task-definition.json \
    --region $REGION \
    --query 'taskDefinition.taskDefinitionArn' \
    --output text)

# Create ECS service
echo "Creating ECS service..."
aws ecs create-service \
    --cluster $CLUSTER_NAME \
    --service-name $SERVICE_NAME \
    --task-definition $TASK_DEFINITION_ARN \
    --desired-count $DESIRED_COUNT \
    --launch-type FARGATE \
    --platform-version LATEST \
    --network-configuration '{
        "awsvpcConfiguration": {
            "subnets": ["'$PRIVATE_SUBNET_ID'"],
            "securityGroups": ["'$ECS_SG_ID'"],
            "assignPublicIp": "DISABLED"
        }
    }' \
    --load-balancers '[
        {
            "targetGroupArn": "'$TARGET_GROUP_ARN'",
            "containerName": "'$APP_NAME'",
            "containerPort": '$CONTAINER_PORT'
        }
    ]' \
    --health-check-grace-period-seconds 120 \
    --deployment-configuration '{
        "deploymentCircuitBreaker": {
            "enable": true,
            "rollback": true
        },
        "maximumPercent": 200,
        "minimumHealthyPercent": 100
    }' \
    --enable-execute-command \
    --region $REGION

# Create CodeBuild project
echo "Creating CodeBuild project..."
cat > deploy/buildspec.yml << EOF
version: 0.2

phases:
  pre_build:
    commands:
      - echo Logging in to Amazon ECR...
      - aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
      - COMMIT_HASH=\$(echo \$CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c 1-7)
      - IMAGE_TAG=\${COMMIT_HASH:=latest}
  build:
    commands:
      - echo Build started on \`date\`
      - echo Building the Docker image...
      - docker build -t $APP_NAME:latest .
      - docker tag $APP_NAME:latest $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:\$IMAGE_TAG
  post_build:
    commands:
      - echo Build completed on \`date\`
      - echo Pushing the Docker image...
      - docker push $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:\$IMAGE_TAG
      - printf '{"ImageURI":"%s"}' $AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$APP_NAME:\$IMAGE_TAG > imageDetail.json
artifacts:
  files:
    - imageDetail.json
    - deploy/task-definition.json
    - deploy/appspec.yaml
EOF

# Create AppSpec file for CodeDeploy
cat > deploy/appspec.yaml << EOF
version: 0.0
Resources:
  - TargetService:
      Type: AWS::ECS::Service
      Properties:
        TaskDefinition: <TASK_DEFINITION>
        LoadBalancerInfo:
          ContainerName: "$APP_NAME"
          ContainerPort: $CONTAINER_PORT
        PlatformVersion: "LATEST"
Hooks:
  - BeforeInstall: "LambdaFunctionToValidateBeforeInstall"
  - AfterInstall: "LambdaFunctionToValidateAfterInstall"
  - AfterAllowTestTraffic: "LambdaFunctionToValidateTestTraffic"
  - BeforeAllowTraffic: "LambdaFunctionToValidateBeforeTraffic"
  - AfterAllowTraffic: "LambdaFunctionToValidateAfterTraffic"
EOF

# Create CodeBuild project
aws codebuild create-project \
    --name "$APP_NAME-build" \
    --source "{
        \"type\": \"CODECOMMIT\",
        \"location\": \"https://git-codecommit.$REGION.amazonaws.com/v1/repos/$APP_NAME\"
    }" \
    --artifacts "{
        \"type\": \"S3\",
        \"location\": \"$APP_NAME-artifacts\"
    }" \
    --environment "{
        \"type\": \"LINUX_CONTAINER\",
        \"image\": \"aws/codebuild/standard:5.0\",
        \"computeType\": \"BUILD_GENERAL1_SMALL\",
        \"privilegedMode\": true,
        \"environmentVariables\": [
            {
                \"name\": \"AWS_DEFAULT_REGION\",
                \"value\": \"$REGION\"
            },
            {
                \"name\": \"AWS_ACCOUNT_ID\",
                \"value\": \"$AWS_ACCOUNT_ID\"
            },
            {
                \"name\": \"IMAGE_REPO_NAME\",
                \"value\": \"$APP_NAME\"
            }
        ]
    }" \
    --service-role "arn:aws:iam::$AWS_ACCOUNT_ID:role/${APP_NAME}BuildRole" \
    --region $REGION

# Create CodePipeline
echo "Creating CodePipeline..."
aws codepipeline create-pipeline \
    --pipeline-name "$APP_NAME-pipeline" \
    --role-arn "arn:aws:iam::$AWS_ACCOUNT_ID:role/${APP_NAME}PipelineRole" \
    --artifact-store "{
        \"type\": \"S3\",
        \"location\": \"$APP_NAME-artifacts\"
    }" \
    --stages '[
        {
            "name": "Source",
            "actions": [
                {
                    "name": "Source",
                    "actionTypeId": {
                        "category": "Source",
                        "owner": "AWS",
                        "provider": "CodeCommit",
                        "version": "1"
                    },
                    "configuration": {
                        "RepositoryName": "'$APP_NAME'",
                        "BranchName": "main"
                    },
                    "outputArtifacts": [
                        {
                            "name": "SourceCode"
                        }
                    ],
                    "runOrder": 1
                }
            ]
        },
        {
            "name": "Build",
            "actions": [
                {
                    "name": "Build",
                    "actionTypeId": {
                        "category": "Build",
                        "owner": "AWS",
                        "provider": "CodeBuild",
                        "version": "1"
                    },
                    "configuration": {
                        "ProjectName": "'$APP_NAME'-build"
                    },
                    "inputArtifacts": [
                        {
                            "name": "SourceCode"
                        }
                    ],
                    "outputArtifacts": [
                        {
                            "name": "BuildOutput"
                        }
                    ],
                    "runOrder": 1
                }
            ]
        },
        {
            "name": "Deploy",
            "actions": [
                {
                    "name": "Deploy",
                    "actionTypeId": {
                        "category": "Deploy",
                        "owner": "AWS",
                        "provider": "ECS",
                        "version": "1"
                    },
                    "configuration": {
                        "ClusterName": "'$CLUSTER_NAME'",
                        "ServiceName": "'$SERVICE_NAME'",
                        "FileName": "imageDetail.json"
                    },
                    "inputArtifacts": [
                        {
                            "name": "BuildOutput"
                        }
                    ],
                    "runOrder": 1
                }
            ]
        }
    ]' \
    --region $REGION

echo "Deployment configuration completed!"
echo "ECR Repository: $APP_NAME"
echo "Task Definition: $TASK_FAMILY"
echo "ECS Service: $SERVICE_NAME"
echo "CodeBuild Project: $APP_NAME-build"
echo "CodePipeline: $APP_NAME-pipeline" 