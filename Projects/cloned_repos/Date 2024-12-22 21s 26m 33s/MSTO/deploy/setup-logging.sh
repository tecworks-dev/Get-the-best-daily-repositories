#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
LOG_GROUP="/ecs/$APP_NAME"
RETENTION_DAYS=30
EXPORT_BUCKET="$APP_NAME-logs"

echo "Setting up logging configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Create S3 bucket for log exports
echo "Creating S3 bucket for log exports..."
aws s3api create-bucket \
    --bucket $EXPORT_BUCKET \
    --region $REGION \
    --create-bucket-configuration LocationConstraint=$REGION

# Enable bucket encryption
aws s3api put-bucket-encryption \
    --bucket $EXPORT_BUCKET \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'

# Set bucket lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
    --bucket $EXPORT_BUCKET \
    --lifecycle-configuration '{
        "Rules": [
            {
                "ID": "ExportedLogsRetention",
                "Status": "Enabled",
                "Prefix": "",
                "Transitions": [
                    {
                        "Days": 30,
                        "StorageClass": "STANDARD_IA"
                    },
                    {
                        "Days": 90,
                        "StorageClass": "GLACIER"
                    }
                ],
                "Expiration": {
                    "Days": 365
                }
            }
        ]
    }'

# Create CloudWatch log group if it doesn't exist
echo "Creating CloudWatch log group..."
aws logs create-log-group \
    --log-group-name $LOG_GROUP \
    --region $REGION || true

# Set log retention
aws logs put-retention-policy \
    --log-group-name $LOG_GROUP \
    --retention-in-days $RETENTION_DAYS \
    --region $REGION

# Create IAM role for log exports
echo "Creating IAM role for log exports..."
LOG_EXPORT_ROLE_ARN=$(aws iam create-role \
    --role-name "${APP_NAME}LogExportRole" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "logs.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }' \
    --query 'Role.Arn' \
    --output text)

# Create and attach policy for log exports
aws iam put-role-policy \
    --role-name "${APP_NAME}LogExportRole" \
    --policy-name "${APP_NAME}LogExportPolicy" \
    --policy-document '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:GetBucketLocation"
                ],
                "Resource": [
                    "arn:aws:s3:::'$EXPORT_BUCKET'",
                    "arn:aws:s3:::'$EXPORT_BUCKET'/*"
                ]
            }
        ]
    }'

# Create log subscription filters
echo "Creating log subscription filters..."

# Error logs filter
aws logs put-subscription-filter \
    --log-group-name $LOG_GROUP \
    --filter-name "${APP_NAME}-errors" \
    --filter-pattern "ERROR" \
    --destination-arn "arn:aws:logs:$REGION:$AWS_ACCOUNT_ID:destination:${APP_NAME}-error-logs" \
    --role-arn $LOG_EXPORT_ROLE_ARN \
    --region $REGION

# Warning logs filter
aws logs put-subscription-filter \
    --log-group-name $LOG_GROUP \
    --filter-name "${APP_NAME}-warnings" \
    --filter-pattern "WARN" \
    --destination-arn "arn:aws:logs:$REGION:$AWS_ACCOUNT_ID:destination:${APP_NAME}-warning-logs" \
    --role-arn $LOG_EXPORT_ROLE_ARN \
    --region $REGION

# Create CloudWatch dashboard for logs
echo "Creating CloudWatch dashboard for logs..."
cat > deploy/logging-dashboard.json << EOF
{
    "widgets": [
        {
            "type": "log",
            "x": 0,
            "y": 0,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '$LOG_GROUP' | fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
                "region": "$REGION",
                "title": "Recent Error Logs",
                "view": "table"
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
                    ["MSTO", "ErrorCount", { "stat": "Sum", "period": 300 }]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Error Count",
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
                    ["MSTO", "WarningCount", { "stat": "Sum", "period": 300 }]
                ],
                "view": "timeSeries",
                "stacked": false,
                "region": "$REGION",
                "title": "Warning Count",
                "period": 300
            }
        },
        {
            "type": "log",
            "x": 0,
            "y": 12,
            "width": 24,
            "height": 6,
            "properties": {
                "query": "SOURCE '$LOG_GROUP' | fields @timestamp, @message\n| filter @message like /Processing time:/\n| parse @message /Processing time: * ms/ as processing_time\n| stats avg(processing_time) as avg_time, max(processing_time) as max_time by bin(5m)",
                "region": "$REGION",
                "title": "Processing Time Analysis",
                "view": "table"
            }
        }
    ]
}
EOF

# Create logging dashboard
aws cloudwatch put-dashboard \
    --dashboard-name "${APP_NAME}-logging" \
    --dashboard-body file://deploy/logging-dashboard.json \
    --region $REGION

# Create CloudWatch alarms for logs
echo "Creating CloudWatch alarms for logs..."

# High error rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-error-rate" \
    --alarm-description "Alert when error rate is high" \
    --metric-name ErrorCount \
    --namespace MSTO \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions "arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts" \
    --region $REGION

# High warning rate alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-high-warning-rate" \
    --alarm-description "Alert when warning rate is high" \
    --metric-name WarningCount \
    --namespace MSTO \
    --statistic Sum \
    --period 300 \
    --threshold 20 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 2 \
    --alarm-actions "arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts" \
    --region $REGION

# Create log insights queries
echo "Creating CloudWatch Logs Insights queries..."

# Save queries to CloudWatch Logs Insights
aws logs put-query-definition \
    --name "${APP_NAME}-error-analysis" \
    --query-string "SOURCE '$LOG_GROUP' | fields @timestamp, @message | filter @message like /ERROR/ | stats count(*) as error_count by bin(5m)" \
    --region $REGION

aws logs put-query-definition \
    --name "${APP_NAME}-performance-analysis" \
    --query-string "SOURCE '$LOG_GROUP' | fields @timestamp, @message | filter @message like /Processing time:/ | parse @message /Processing time: * ms/ as processing_time | stats avg(processing_time) as avg_time, max(processing_time) as max_time by bin(5m)" \
    --region $REGION

aws logs put-query-definition \
    --name "${APP_NAME}-strategy-signals" \
    --query-string "SOURCE '$LOG_GROUP' | fields @timestamp, @message | filter @message like /Signal generated/ | parse @message /strategy: *, ticker: */ as strategy, ticker | stats count(*) as signal_count by strategy, ticker, bin(1h)" \
    --region $REGION

echo "Logging configuration completed!"
echo "Log group: $LOG_GROUP"
echo "Log retention: $RETENTION_DAYS days"
echo "Export bucket: $EXPORT_BUCKET"
echo "CloudWatch dashboard: ${APP_NAME}-logging" 