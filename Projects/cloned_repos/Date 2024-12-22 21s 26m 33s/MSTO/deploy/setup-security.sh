#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
KMS_ALIAS="alias/$APP_NAME"
WAF_LIMIT=2000
WAF_BLOCK_PERIOD=240

echo "Setting up security configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Create KMS key
echo "Creating KMS key..."
KMS_KEY_ID=$(aws kms create-key \
    --description "Key for $APP_NAME encryption" \
    --tags TagKey=Application,TagValue=$APP_NAME \
    --region $REGION \
    --query 'KeyMetadata.KeyId' \
    --output text)

# Create alias for KMS key
aws kms create-alias \
    --alias-name $KMS_ALIAS \
    --target-key-id $KMS_KEY_ID \
    --region $REGION

# Create WAF web ACL
echo "Creating WAF web ACL..."
WEB_ACL_ID=$(aws wafv2 create-web-acl \
    --name "$APP_NAME-web-acl" \
    --scope REGIONAL \
    --default-action Block={} \
    --description "Web ACL for $APP_NAME" \
    --rules '[
        {
            "Name": "AWSManagedRulesCommonRuleSet",
            "Priority": 1,
            "Statement": {
                "ManagedRuleGroupStatement": {
                    "VendorName": "AWS",
                    "Name": "AWSManagedRulesCommonRuleSet"
                }
            },
            "OverrideAction": {
                "None": {}
            },
            "VisibilityConfig": {
                "SampledRequestsEnabled": true,
                "CloudWatchMetricsEnabled": true,
                "MetricName": "AWSManagedRulesCommonRuleSetMetric"
            }
        },
        {
            "Name": "RateLimit",
            "Priority": 2,
            "Statement": {
                "RateBasedStatement": {
                    "Limit": '$WAF_LIMIT',
                    "AggregateKeyType": "IP"
                }
            },
            "Action": {
                "Block": {
                    "CustomResponse": {
                        "ResponseCode": 429,
                        "CustomResponseBodyKey": "TooManyRequests"
                    }
                }
            },
            "VisibilityConfig": {
                "SampledRequestsEnabled": true,
                "CloudWatchMetricsEnabled": true,
                "MetricName": "RateLimitMetric"
            }
        }
    ]' \
    --visibility-config '{
        "SampledRequestsEnabled": true,
        "CloudWatchMetricsEnabled": true,
        "MetricName": "WebACLMetric"
    }' \
    --region $REGION \
    --query 'Summary.Id' \
    --output text)

# Associate WAF web ACL with ALB
aws wafv2 associate-web-acl \
    --web-acl-arn "arn:aws:wafv2:$REGION:$AWS_ACCOUNT_ID:regional/webacl/$APP_NAME-web-acl/$WEB_ACL_ID" \
    --resource-arn $ALB_ARN \
    --region $REGION

# Create SecurityHub standards subscription
echo "Enabling SecurityHub standards..."
aws securityhub enable-security-hub \
    --enable-default-standards \
    --control-finding-generator SECURITY_CONTROL \
    --region $REGION

# Enable GuardDuty
echo "Enabling GuardDuty..."
aws guardduty create-detector \
    --enable \
    --finding-publishing-frequency FIFTEEN_MINUTES \
    --region $REGION

# Create IAM password policy
echo "Setting up IAM password policy..."
aws iam update-account-password-policy \
    --minimum-password-length 14 \
    --require-symbols \
    --require-numbers \
    --require-uppercase-characters \
    --require-lowercase-characters \
    --allow-users-to-change-password \
    --max-password-age 90 \
    --password-reuse-prevention 24 \
    --hard-expiry

# Create S3 bucket for security logs
echo "Creating security logs bucket..."
SECURITY_LOGS_BUCKET="$APP_NAME-security-logs"
aws s3api create-bucket \
    --bucket $SECURITY_LOGS_BUCKET \
    --region $REGION \
    --create-bucket-configuration LocationConstraint=$REGION

# Enable S3 bucket encryption
aws s3api put-bucket-encryption \
    --bucket $SECURITY_LOGS_BUCKET \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "aws:kms",
                    "KMSMasterKeyID": "'$KMS_KEY_ID'"
                }
            }
        ]
    }'

# Enable S3 bucket versioning
aws s3api put-bucket-versioning \
    --bucket $SECURITY_LOGS_BUCKET \
    --versioning-configuration Status=Enabled

# Set S3 bucket policy
aws s3api put-bucket-policy \
    --bucket $SECURITY_LOGS_BUCKET \
    --policy '{
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyUnencryptedObjectUploads",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:PutObject",
                "Resource": "arn:aws:s3:::'$SECURITY_LOGS_BUCKET'/*",
                "Condition": {
                    "StringNotEquals": {
                        "s3:x-amz-server-side-encryption": "aws:kms"
                    }
                }
            },
            {
                "Sid": "DenyHTTP",
                "Effect": "Deny",
                "Principal": "*",
                "Action": "s3:*",
                "Resource": "arn:aws:s3:::'$SECURITY_LOGS_BUCKET'/*",
                "Condition": {
                    "Bool": {
                        "aws:SecureTransport": "false"
                    }
                }
            }
        ]
    }'

# Enable CloudTrail
echo "Enabling CloudTrail..."
aws cloudtrail create-trail \
    --name "$APP_NAME-trail" \
    --s3-bucket-name $SECURITY_LOGS_BUCKET \
    --is-multi-region-trail \
    --enable-log-file-validation \
    --kms-key-id $KMS_KEY_ID \
    --cloud-watch-logs-log-group-arn "arn:aws:logs:$REGION:$AWS_ACCOUNT_ID:log-group:/aws/cloudtrail/$APP_NAME:*" \
    --cloud-watch-logs-role-arn "arn:aws:iam::$AWS_ACCOUNT_ID:role/${APP_NAME}CloudTrailRole" \
    --region $REGION

aws cloudtrail start-logging \
    --name "$APP_NAME-trail" \
    --region $REGION

# Create CloudWatch alarms for security events
echo "Creating security alarms..."

# Root account usage alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-root-account-usage" \
    --alarm-description "Alert when root account is used" \
    --metric-name RootAccountUsage \
    --namespace AWS/IAM \
    --statistic Sum \
    --period 300 \
    --threshold 1 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions "arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts" \
    --region $REGION

# Unauthorized API calls alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-unauthorized-api-calls" \
    --alarm-description "Alert on unauthorized API calls" \
    --metric-name UnauthorizedAttemptCount \
    --namespace AWS/CloudTrail \
    --statistic Sum \
    --period 300 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions "arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts" \
    --region $REGION

# Create Config rules
echo "Setting up AWS Config rules..."
aws configservice put-configuration-recorder \
    --configuration-recorder name=$APP_NAME-recorder,roleARN=arn:aws:iam::$AWS_ACCOUNT_ID:role/${APP_NAME}ConfigRole \
    --recording-group allSupported=true,includeGlobalResources=true \
    --region $REGION

aws configservice put-config-rule \
    --config-rule '{
        "ConfigRuleName": "'$APP_NAME'-encrypted-volumes",
        "Source": {
            "Owner": "AWS",
            "SourceIdentifier": "ENCRYPTED_VOLUMES"
        },
        "Scope": {
            "ComplianceResourceTypes": [
                "AWS::EC2::Volume"
            ]
        }
    }' \
    --region $REGION

aws configservice put-config-rule \
    --config-rule '{
        "ConfigRuleName": "'$APP_NAME'-rds-encryption",
        "Source": {
            "Owner": "AWS",
            "SourceIdentifier": "RDS_STORAGE_ENCRYPTED"
        },
        "Scope": {
            "ComplianceResourceTypes": [
                "AWS::RDS::DBInstance"
            ]
        }
    }' \
    --region $REGION

aws configservice start-configuration-recorder \
    --configuration-recorder-name $APP_NAME-recorder \
    --region $REGION

echo "Security configuration completed!"
echo "KMS Key ID: $KMS_KEY_ID"
echo "WAF Web ACL ID: $WEB_ACL_ID"
echo "Security Logs Bucket: $SECURITY_LOGS_BUCKET"
echo "CloudTrail: $APP_NAME-trail" 