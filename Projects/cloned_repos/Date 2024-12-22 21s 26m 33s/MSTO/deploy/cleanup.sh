#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
DASHBOARD_NAME="$APP_NAME-dashboard"
SNS_TOPIC_NAME="$APP_NAME-alerts"
LOG_GROUP_NAME="/ecs/$APP_NAME"
CLUSTER_NAME="$APP_NAME-cluster"
SERVICE_NAME="$APP_NAME-service"
TASK_FAMILY="$APP_NAME-task"

echo "Cleaning up AWS resources for $APP_NAME..."

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SNS_TOPIC_ARN="arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$SNS_TOPIC_NAME"

# Delete ECS service
echo "Deleting ECS service..."
aws ecs update-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --desired-count 0 \
    --region $REGION || true

aws ecs delete-service \
    --cluster $CLUSTER_NAME \
    --service $SERVICE_NAME \
    --region $REGION || true

# Wait for service to be deleted
echo "Waiting for service to be deleted..."
aws ecs wait services-inactive \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION || true

# Delete ECS cluster
echo "Deleting ECS cluster..."
aws ecs delete-cluster \
    --cluster $CLUSTER_NAME \
    --region $REGION || true

# Delete task definitions
echo "Deleting task definitions..."
for arn in $(aws ecs list-task-definitions \
    --family-prefix $TASK_FAMILY \
    --region $REGION \
    --query 'taskDefinitionArns[]' \
    --output text); do
    aws ecs deregister-task-definition \
        --task-definition $arn \
        --region $REGION || true
done

# Delete ECR repository
echo "Deleting ECR repository..."
aws ecr delete-repository \
    --repository-name $APP_NAME \
    --force \
    --region $REGION || true

# Delete CloudWatch dashboard
echo "Deleting CloudWatch dashboard..."
aws cloudwatch delete-dashboards \
    --dashboard-names $DASHBOARD_NAME \
    --region $REGION || true

# Delete CloudWatch alarms
echo "Deleting CloudWatch alarms..."
aws cloudwatch describe-alarms \
    --alarm-name-prefix $APP_NAME \
    --region $REGION \
    --query 'MetricAlarms[].AlarmName' \
    --output text | tr '\t' '\n' | while read alarm; do
    aws cloudwatch delete-alarms \
        --alarm-names "$alarm" \
        --region $REGION || true
done

# Delete metric filters
echo "Deleting metric filters..."
aws logs describe-metric-filters \
    --log-group-name $LOG_GROUP_NAME \
    --region $REGION \
    --query 'metricFilters[].filterName' \
    --output text | tr '\t' '\n' | while read filter; do
    aws logs delete-metric-filter \
        --log-group-name $LOG_GROUP_NAME \
        --filter-name "$filter" \
        --region $REGION || true
done

# Delete log group
echo "Deleting CloudWatch log group..."
aws logs delete-log-group \
    --log-group-name $LOG_GROUP_NAME \
    --region $REGION || true

# Delete SNS topic
echo "Deleting SNS topic..."
aws sns delete-topic \
    --topic-arn $SNS_TOPIC_ARN \
    --region $REGION || true

# Delete EFS filesystem
echo "Deleting EFS filesystem..."
# First delete mount targets
for mt in $(aws efs describe-mount-targets \
    --file-system-id $EFS_FILESYSTEM_ID \
    --region $REGION \
    --query 'MountTargets[].MountTargetId' \
    --output text); do
    aws efs delete-mount-target \
        --mount-target-id $mt \
        --region $REGION || true
done

# Wait for mount targets to be deleted
sleep 30

# Delete filesystem
aws efs delete-file-system \
    --file-system-id $EFS_FILESYSTEM_ID \
    --region $REGION || true

# Delete security groups
echo "Deleting security groups..."
for sg in $(aws ec2 describe-security-groups \
    --filters "Name=group-name,Values=$APP_NAME-*" \
    --query 'SecurityGroups[].GroupId' \
    --output text); do
    aws ec2 delete-security-group \
        --group-id $sg \
        --region $REGION || true
done

echo "Cleanup completed!" 