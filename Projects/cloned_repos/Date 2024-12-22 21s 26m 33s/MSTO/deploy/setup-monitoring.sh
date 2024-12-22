#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
CLUSTER_NAME="$APP_NAME-cluster"
SERVICE_NAME="$APP_NAME-service"
LOG_GROUP="/ecs/$APP_NAME"
RETENTION_DAYS=30

echo "Setting up monitoring configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Create SNS topic for alerts if it doesn't exist
echo "Creating SNS topic for alerts..."
SNS_TOPIC_ARN=$(aws sns create-topic \
    --name "$APP_NAME-alerts" \
    --region $REGION \
    --query 'TopicArn' \
    --output text)

# Create CloudWatch log group
echo "Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name $LOG_GROUP \
    --region $REGION

# Set log retention
aws logs put-retention-policy \
    --log-group-name $LOG_GROUP \
    --retention-in-days $RETENTION_DAYS \
    --region $REGION

# Create CloudWatch dashboard
echo "Creating CloudWatch dashboard..."
cat > deploy/monitoring-dashboard.json << EOF
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
                "title": "Container Resource Utilization",
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
                    ["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", "$APP_NAME-lb"],
                    [".", "RequestCount", ".", "."],
                    [".", "HTTPCode_Target_5XX_Count", ".", "."],
                    [".", "HTTPCode_Target_4XX_Count", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Application Performance",
                "period": 300
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 6,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '$LOG_GROUP' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
                "region": "$REGION",
                "title": "Error Logs",
                "view": "table"
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["MSTO", "SignalsGenerated", "Strategy", "FundamentalEventDriven"],
                    [".", ".", ".", "SimpleVolatility"]
                ],
                "view": "timeSeries",
                "stacked": true,
                "region": "$REGION",
                "title": "Trading Signals",
                "period": 300
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 12,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["MSTO", "ProcessingLatency", "Component", "DataFetcher"],
                    [".", ".", ".", "Analytics"],
                    [".", ".", ".", "Strategy"]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Component Latencies",
                "period": 300
            }
        }
    ]
}
EOF

# Create monitoring dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "${APP_NAME}-monitoring" \
    --dashboard-body file://deploy/monitoring-dashboard.json \
    --region $REGION

# Create CloudWatch alarms
echo "Creating CloudWatch alarms..."

# High CPU utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-cpu" \
    --alarm-description "Alert when CPU utilization is high" \
    --metric-name CPUUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=ClusterName,Value=$CLUSTER_NAME Name=ServiceName,Value=$SERVICE_NAME \
    --region $REGION

# High memory utilization alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-memory" \
    --alarm-description "Alert when memory utilization is high" \
    --metric-name MemoryUtilization \
    --namespace AWS/ECS \
    --statistic Average \
    --period 300 \
    --threshold 85 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=ClusterName,Value=$CLUSTER_NAME Name=ServiceName,Value=$SERVICE_NAME \
    --region $REGION

# High error rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-error-rate" \
    --alarm-description "Alert when error rate is high" \
    --metric-name HTTPCode_Target_5XX_Count \
    --namespace AWS/ApplicationELB \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=LoadBalancer,Value=$APP_NAME-lb \
    --region $REGION

# High latency alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-latency" \
    --alarm-description "Alert when response time is high" \
    --metric-name TargetResponseTime \
    --namespace AWS/ApplicationELB \
    --statistic Average \
    --period 300 \
    --threshold 5 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=LoadBalancer,Value=$APP_NAME-lb \
    --region $REGION

# Create log metric filters
echo "Creating log metric filters..."

# Error log metric filter
aws logs put-metric-filter \
    --log-group-name $LOG_GROUP \
    --filter-name "${APP_NAME}-errors" \
    --filter-pattern "ERROR" \
    --metric-transformations \
        metricName=ErrorCount,metricNamespace=MSTO,metricValue=1 \
    --region $REGION

# Warning log metric filter
aws logs put-metric-filter \
    --log-group-name $LOG_GROUP \
    --filter-name "${APP_NAME}-warnings" \
    --filter-pattern "WARN" \
    --metric-transformations \
        metricName=WarningCount,metricNamespace=MSTO,metricValue=1 \
    --region $REGION

# Signal generation metric filter
aws logs put-metric-filter \
    --log-group-name $LOG_GROUP \
    --filter-pattern "[timestamp, level, component=Strategy*, event=SignalGenerated]" \
    --filter-name "${APP_NAME}-signals" \
    --metric-transformations \
        metricName=SignalsGenerated,metricNamespace=MSTO,metricValue=1 \
    --region $REGION

# Create composite alarms
echo "Creating composite alarms..."

# Service health composite alarm
aws cloudwatch put-composite-alarm \
    --alarm-name "$APP_NAME-service-health" \
    --alarm-description "Composite alarm for overall service health" \
    --alarm-rule "ALARM(\"$APP_NAME-high-cpu\") OR ALARM(\"$APP_NAME-high-memory\") OR ALARM(\"$APP_NAME-high-error-rate\") OR ALARM(\"$APP_NAME-high-latency\")" \
    --alarm-actions $SNS_TOPIC_ARN \
    --region $REGION

echo "Monitoring configuration completed!"
echo "CloudWatch dashboard: ${APP_NAME}-monitoring"
echo "Log group: $LOG_GROUP"
echo "Log retention: $RETENTION_DAYS days"
echo "SNS topic ARN: $SNS_TOPIC_ARN" 