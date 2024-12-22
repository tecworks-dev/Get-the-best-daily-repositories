#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
BACKUP_VAULT_NAME="$APP_NAME-vault"
BACKUP_PLAN_NAME="$APP_NAME-backup-plan"
IAM_ROLE_NAME="${APP_NAME}BackupRole"

echo "Setting up AWS Backup configuration for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Create backup vault
echo "Creating backup vault..."
aws backup create-backup-vault \
    --backup-vault-name $BACKUP_VAULT_NAME \
    --region $REGION

# Create IAM role for AWS Backup
echo "Creating IAM role for AWS Backup..."
aws iam create-role \
    --role-name $IAM_ROLE_NAME \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "backup.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }'

# Attach necessary policies to the IAM role
aws iam attach-role-policy \
    --role-name $IAM_ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForBackup

aws iam attach-role-policy \
    --role-name $IAM_ROLE_NAME \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSBackupServiceRolePolicyForRestores

# Create backup plan
echo "Creating backup plan..."
BACKUP_PLAN_ID=$(aws backup create-backup-plan \
    --backup-plan '{
        "BackupPlanName": "'$BACKUP_PLAN_NAME'",
        "Rules": [
            {
                "RuleName": "DailyBackups",
                "TargetBackupVaultName": "'$BACKUP_VAULT_NAME'",
                "ScheduleExpression": "cron(0 5 ? * * *)",
                "StartWindowMinutes": 60,
                "CompletionWindowMinutes": 120,
                "Lifecycle": {
                    "MoveToColdStorageAfterDays": 30,
                    "DeleteAfterDays": 365
                },
                "RecoveryPointTags": {
                    "Application": "'$APP_NAME'",
                    "Environment": "Production"
                }
            },
            {
                "RuleName": "WeeklyBackups",
                "TargetBackupVaultName": "'$BACKUP_VAULT_NAME'",
                "ScheduleExpression": "cron(0 5 ? * SAT *)",
                "StartWindowMinutes": 60,
                "CompletionWindowMinutes": 120,
                "Lifecycle": {
                    "MoveToColdStorageAfterDays": 90,
                    "DeleteAfterDays": 730
                },
                "RecoveryPointTags": {
                    "Application": "'$APP_NAME'",
                    "Environment": "Production"
                }
            }
        ]
    }' \
    --region $REGION \
    --query 'BackupPlanId' \
    --output text)

# Create backup selection
echo "Creating backup selection..."
aws backup create-backup-selection \
    --backup-plan-id $BACKUP_PLAN_ID \
    --backup-selection '{
        "SelectionName": "'$APP_NAME'-resources",
        "IamRoleArn": "arn:aws:iam::'$AWS_ACCOUNT_ID':role/'$IAM_ROLE_NAME'",
        "Resources": [
            "arn:aws:rds:'$REGION':'$AWS_ACCOUNT_ID':db:'$APP_NAME'-db",
            "arn:aws:elasticfilesystem:'$REGION':'$AWS_ACCOUNT_ID':file-system/'$EFS_FILESYSTEM_ID'"
        ],
        "ListOfTags": [
            {
                "ConditionType": "STRINGEQUALS",
                "ConditionKey": "Application",
                "ConditionValue": "'$APP_NAME'"
            }
        ]
    }' \
    --region $REGION

# Create notification configuration
echo "Setting up backup notifications..."
SNS_TOPIC_ARN="arn:aws:sns:$REGION:$AWS_ACCOUNT_ID:$APP_NAME-alerts"

# Create event notifications for backup jobs
aws backup put-backup-vault-notifications \
    --backup-vault-name $BACKUP_VAULT_NAME \
    --sns-topic-arn $SNS_TOPIC_ARN \
    --backup-vault-events BACKUP_JOB_STARTED BACKUP_JOB_COMPLETED BACKUP_JOB_FAILED \
    --region $REGION

# Create CloudWatch alarms for backup monitoring
echo "Creating CloudWatch alarms for backup monitoring..."

# Backup job failure alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-backup-job-failed" \
    --alarm-description "Alert when backup job fails" \
    --metric-name BackupJobsFailed \
    --namespace AWS/Backup \
    --statistic Sum \
    --period 300 \
    --threshold 0 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=BackupVault,Value=$BACKUP_VAULT_NAME \
    --region $REGION

# Restore job failure alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "$APP_NAME-restore-job-failed" \
    --alarm-description "Alert when restore job fails" \
    --metric-name RestoreJobsFailed \
    --namespace AWS/Backup \
    --statistic Sum \
    --period 300 \
    --threshold 0 \
    --comparison-operator GreaterThanThreshold \
    --evaluation-periods 1 \
    --alarm-actions $SNS_TOPIC_ARN \
    --dimensions Name=BackupVault,Value=$BACKUP_VAULT_NAME \
    --region $REGION

echo "Backup configuration completed!"
echo "Backup vault: $BACKUP_VAULT_NAME"
echo "Backup plan ID: $BACKUP_PLAN_ID"
echo "Daily backups scheduled for 5:00 AM UTC"
echo "Weekly backups scheduled for Saturday 5:00 AM UTC" 