#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
CLUSTER_NAME="$APP_NAME-cluster"
SERVICE_NAME="$APP_NAME-service"
MIN_CAPACITY=1
MAX_CAPACITY=4
TARGET_CPU_UTILIZATION=70
TARGET_MEMORY_UTILIZATION=70
SCALE_IN_COOLDOWN=300
SCALE_OUT_COOLDOWN=60

echo "Setting up auto-scaling configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Get service ARN
SERVICE_ARN=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services $SERVICE_NAME \
    --region $REGION \
    --query 'services[0].serviceArn' \
    --output text)

# Register scalable target
echo "Registering scalable target..."
aws application-autoscaling register-scalable-target \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --min-capacity $MIN_CAPACITY \
    --max-capacity $MAX_CAPACITY \
    --region $REGION

# Create CPU utilization scaling policy
echo "Creating CPU utilization scaling policy..."
aws application-autoscaling put-scaling-policy \
    --policy-name "${APP_NAME}-cpu-scaling" \
    --policy-type TargetTrackingScaling \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": '$TARGET_CPU_UTILIZATION',
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
        },
        "ScaleOutCooldown": '$SCALE_OUT_COOLDOWN',
        "ScaleInCooldown": '$SCALE_IN_COOLDOWN'
    }' \
    --region $REGION

# Create memory utilization scaling policy
echo "Creating memory utilization scaling policy..."
aws application-autoscaling put-scaling-policy \
    --policy-name "${APP_NAME}-memory-scaling" \
    --policy-type TargetTrackingScaling \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": '$TARGET_MEMORY_UTILIZATION',
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "ECSServiceAverageMemoryUtilization"
        },
        "ScaleOutCooldown": '$SCALE_OUT_COOLDOWN',
        "ScaleInCooldown": '$SCALE_IN_COOLDOWN'
    }' \
    --region $REGION

# Create custom metric scaling policy for signal processing
echo "Creating signal processing scaling policy..."
aws application-autoscaling put-scaling-policy \
    --policy-name "${APP_NAME}-signal-scaling" \
    --policy-type TargetTrackingScaling \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --target-tracking-scaling-policy-configuration '{
        "TargetValue": 100,
        "CustomizedMetricSpecification": {
            "MetricName": "ProcessingTime",
            "Namespace": "MSTO",
            "Dimensions": [
                {
                    "Name": "Service",
                    "Value": "msto"
                }
            ],
            "Statistic": "Average",
            "Unit": "Milliseconds"
        },
        "ScaleOutCooldown": '$SCALE_OUT_COOLDOWN',
        "ScaleInCooldown": '$SCALE_IN_COOLDOWN'
    }' \
    --region $REGION

# Create scheduled scaling for market hours
echo "Creating scheduled scaling for market hours..."

# Scale up before market open (30 minutes before)
aws application-autoscaling put-scheduled-action \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --scheduled-action-name "${APP_NAME}-market-open" \
    --schedule "cron(30 13 ? * MON-FRI *)" \
    --scalable-target-action MinCapacity=$MIN_CAPACITY,MaxCapacity=$MAX_CAPACITY \
    --region $REGION

# Scale down after market close (30 minutes after)
aws application-autoscaling put-scheduled-action \
    --service-namespace ecs \
    --scalable-dimension ecs:service:DesiredCount \
    --resource-id service/$CLUSTER_NAME/$SERVICE_NAME \
    --scheduled-action-name "${APP_NAME}-market-close" \
    --schedule "cron(30 20 ? * MON-FRI *)" \
    --scalable-target-action MinCapacity=1,MaxCapacity=1 \
    --region $REGION

# Create CloudWatch dashboard for scaling metrics
echo "Creating CloudWatch dashboard for scaling metrics..."
cat > deploy/scaling-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/ECS", "CPUUtilization", "ServiceName", "$SERVICE_NAME", "ClusterName", "$CLUSTER_NAME"],
                    [".", "MemoryUtilization", ".", ".", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Service Resource Utilization",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/ApplicationELB", "RequestCount", "TargetGroup", "$APP_NAME-target-group"],
                    [".", "TargetResponseTime", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Service Performance",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/ECS", "DesiredCount", "ServiceName", "$SERVICE_NAME", "ClusterName", "$CLUSTER_NAME"],
                    [".", "RunningTaskCount", ".", ".", ".", "."],
                    [".", "PendingTaskCount", ".", ".", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Service Scaling",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["MSTO", "ProcessingTime", "Service", "msto", { "stat": "Average" }],
                    ["...", { "stat": "p90" }],
                    ["...", { "stat": "Maximum" }]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Signal Processing Time",
                "period": 300
            }
        }
    ]
}
EOF

# Create scaling dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "${APP_NAME}-scaling" \
    --dashboard-body file://deploy/scaling-dashboard.json \
    --region $REGION

# Create CloudWatch alarms for scaling events
echo "Creating CloudWatch alarms for scaling events..."

# Service scaling alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-scaling-alarm" \
    --alarm-description "Alert on service scaling events" \
    --metric-name DesiredCount \
    --namespace AWS/ECS \
    --statistic Maximum \
    --period 300 \
    --threshold $((MAX_CAPACITY - 1)) \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions "arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts" \
    --dimensions Name=ClusterName,Value=$CLUSTER_NAME Name=ServiceName,Value=$SERVICE_NAME \
    --region $REGION

echo "Auto-scaling configuration completed!"
echo "Service will scale between $MIN_CAPACITY and $MAX_CAPACITY tasks"
echo "CPU target utilization: $TARGET_CPU_UTILIZATION%"
echo "Memory target utilization: $TARGET_MEMORY_UTILIZATION%"
echo "Market hours scaling: 9:30 AM - 4:00 PM ET" 