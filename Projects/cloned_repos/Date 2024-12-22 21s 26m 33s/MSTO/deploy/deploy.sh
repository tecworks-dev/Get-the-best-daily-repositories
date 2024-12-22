#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
ECR_REPO="$APP_NAME"
CLUSTER_NAME="$APP_NAME-cluster"
SERVICE_NAME="$APP_NAME-service"
TASK_FAMILY="$APP_NAME-task"
CONTAINER_NAME="$APP_NAME"
CONTAINER_PORT=8080
DESIRED_COUNT=1

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO_URI="$AWS_ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO"

echo "Deploying $APP_NAME to AWS ECS..."

# Build and tag Docker image
echo "Building Docker image..."
docker build -t $APP_NAME:latest .
docker tag $APP_NAME:latest $ECR_REPO_URI:latest

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_REPO_URI

# Push image to ECR
echo "Pushing image to ECR..."
docker push $ECR_REPO_URI:latest

# Update ECS task definition
echo "Updating ECS task definition..."
TASK_DEFINITION=$(aws ecs describe-task-definition --task-definition $TASK_FAMILY --region $REGION)
NEW_TASK_DEFINITION=$(echo $TASK_DEFINITION | jq --arg IMAGE "$ECR_REPO_URI:latest" \
  '.taskDefinition | .containerDefinitions[0].image = $IMAGE | del(.taskDefinitionArn) | del(.revision) | del(.status) | del(.requiresAttributes) | del(.compatibilities)')

# Register new task definition
NEW_TASK_INFO=$(aws ecs register-task-definition --region $REGION --cli-input-json "$NEW_TASK_DEFINITION")
NEW_REVISION=$(echo $NEW_TASK_INFO | jq '.taskDefinition.revision')

# Update service with new task definition
echo "Updating ECS service..."
aws ecs update-service \
  --region $REGION \
  --cluster $CLUSTER_NAME \
  --service $SERVICE_NAME \
  --task-definition "$TASK_FAMILY:$NEW_REVISION" \
  --desired-count $DESIRED_COUNT

echo "Deployment completed successfully!"

# Wait for service to stabilize
echo "Waiting for service to stabilize..."
aws ecs wait services-stable \
  --region $REGION \
  --cluster $CLUSTER_NAME \
  --services $SERVICE_NAME

echo "Service is stable. Deployment finished!"

# Print service URL
TASK_ARN=$(aws ecs list-tasks --cluster $CLUSTER_NAME --service-name $SERVICE_NAME --query 'taskArns[0]' --output text)
ENI_ID=$(aws ecs describe-tasks --cluster $CLUSTER_NAME --tasks $TASK_ARN --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text)
PUBLIC_IP=$(aws ec2 describe-network-interfaces --network-interface-ids $ENI_ID --query 'NetworkInterfaces[0].Association.PublicIp' --output text)

echo "Service is available at: http://$PUBLIC_IP:$CONTAINER_PORT"
echo "Health check endpoint: http://$PUBLIC_IP:$CONTAINER_PORT/health"