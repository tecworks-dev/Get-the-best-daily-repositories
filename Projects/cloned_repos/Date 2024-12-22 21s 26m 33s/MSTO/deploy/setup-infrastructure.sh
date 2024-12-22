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
DB_INSTANCE_CLASS="db.t3.micro"
DB_ALLOCATED_STORAGE=20

echo "Setting up infrastructure for $APP_NAME..."

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create VPC
echo "Creating VPC..."
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=$APP_NAME-vpc}]" \
    --region $REGION \
    --query 'Vpc.VpcId' \
    --output text)

# Enable DNS hostnames
aws ec2 modify-vpc-attribute \
    --vpc-id $VPC_ID \
    --enable-dns-hostnames \
    --region $REGION

# Create internet gateway
echo "Creating internet gateway..."
IGW_ID=$(aws ec2 create-internet-gateway \
    --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=$APP_NAME-igw}]" \
    --region $REGION \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

# Attach internet gateway to VPC
aws ec2 attach-internet-gateway \
    --vpc-id $VPC_ID \
    --internet-gateway-id $IGW_ID \
    --region $REGION

# Create public subnet
echo "Creating public subnet..."
PUBLIC_SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone "${REGION}a" \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$APP_NAME-public-subnet}]" \
    --region $REGION \
    --query 'Subnet.SubnetId' \
    --output text)

# Create private subnet
echo "Creating private subnet..."
PRIVATE_SUBNET_ID=$(aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.2.0/24 \
    --availability-zone "${REGION}b" \
    --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=$APP_NAME-private-subnet}]" \
    --region $REGION \
    --query 'Subnet.SubnetId' \
    --output text)

# Create route table for public subnet
echo "Creating route tables..."
PUBLIC_RT_ID=$(aws ec2 create-route-table \
    --vpc-id $VPC_ID \
    --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=$APP_NAME-public-rt}]" \
    --region $REGION \
    --query 'RouteTable.RouteTableId' \
    --output text)

# Create route to internet gateway
aws ec2 create-route \
    --route-table-id $PUBLIC_RT_ID \
    --destination-cidr-block 0.0.0.0/0 \
    --gateway-id $IGW_ID \
    --region $REGION

# Associate public subnet with route table
aws ec2 associate-route-table \
    --subnet-id $PUBLIC_SUBNET_ID \
    --route-table-id $PUBLIC_RT_ID \
    --region $REGION

# Create security group for ECS tasks
echo "Creating security groups..."
ECS_SG_ID=$(aws ec2 create-security-group \
    --group-name "$APP_NAME-ecs-sg" \
    --description "Security group for ECS tasks" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Add inbound rules for ECS security group
aws ec2 authorize-security-group-ingress \
    --group-id $ECS_SG_ID \
    --protocol tcp \
    --port $CONTAINER_PORT \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Create security group for RDS
RDS_SG_ID=$(aws ec2 create-security-group \
    --group-name "$APP_NAME-rds-sg" \
    --description "Security group for RDS" \
    --vpc-id $VPC_ID \
    --region $REGION \
    --query 'GroupId' \
    --output text)

# Add inbound rule for RDS from ECS security group
aws ec2 authorize-security-group-ingress \
    --group-id $RDS_SG_ID \
    --protocol tcp \
    --port 5432 \
    --source-group $ECS_SG_ID \
    --region $REGION

# Create ECS cluster
echo "Creating ECS cluster..."
aws ecs create-cluster \
    --cluster-name $CLUSTER_NAME \
    --capacity-providers FARGATE \
    --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
    --region $REGION

# Create task execution role
echo "Creating IAM roles..."
TASK_EXECUTION_ROLE_ARN=$(aws iam create-role \
    --role-name "${APP_NAME}TaskExecutionRole" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }' \
    --query 'Role.Arn' \
    --output text)

# Attach task execution role policy
aws iam attach-role-policy \
    --role-name "${APP_NAME}TaskExecutionRole" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create task role
TASK_ROLE_ARN=$(aws iam create-role \
    --role-name "${APP_NAME}TaskRole" \
    --assume-role-policy-document '{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }]
    }' \
    --query 'Role.Arn' \
    --output text)

# Attach necessary policies to task role
aws iam attach-role-policy \
    --role-name "${APP_NAME}TaskRole" \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Create RDS subnet group
echo "Creating RDS subnet group..."
aws rds create-db-subnet-group \
    --db-subnet-group-name "$APP_NAME-db-subnet-group" \
    --db-subnet-group-description "Subnet group for $APP_NAME RDS" \
    --subnet-ids $PRIVATE_SUBNET_ID $PUBLIC_SUBNET_ID \
    --region $REGION

# Create RDS instance
echo "Creating RDS instance..."
DB_INSTANCE_IDENTIFIER="$APP_NAME-db"
aws rds create-db-instance \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --db-instance-class $DB_INSTANCE_CLASS \
    --engine postgres \
    --master-username "msto_admin" \
    --master-user-password "$(openssl rand -base64 32)" \
    --allocated-storage $DB_ALLOCATED_STORAGE \
    --vpc-security-group-ids $RDS_SG_ID \
    --db-subnet-group-name "$APP_NAME-db-subnet-group" \
    --no-publicly-accessible \
    --region $REGION

# Create EFS file system
echo "Creating EFS file system..."
EFS_FILESYSTEM_ID=$(aws efs create-file-system \
    --performance-mode generalPurpose \
    --throughput-mode bursting \
    --encrypted \
    --tags Key=Name,Value=$APP_NAME-efs \
    --region $REGION \
    --query 'FileSystemId' \
    --output text)

# Create EFS mount targets
aws efs create-mount-target \
    --file-system-id $EFS_FILESYSTEM_ID \
    --subnet-id $PRIVATE_SUBNET_ID \
    --security-groups $ECS_SG_ID \
    --region $REGION

# Create Application Load Balancer
echo "Creating Application Load Balancer..."
ALB_ARN=$(aws elbv2 create-load-balancer \
    --name "$APP_NAME-lb" \
    --subnets $PUBLIC_SUBNET_ID $PRIVATE_SUBNET_ID \
    --security-groups $ECS_SG_ID \
    --region $REGION \
    --query 'LoadBalancers[0].LoadBalancerArn' \
    --output text)

# Create target group
TARGET_GROUP_ARN=$(aws elbv2 create-target-group \
    --name "$APP_NAME-target-group" \
    --protocol HTTP \
    --port $CONTAINER_PORT \
    --vpc-id $VPC_ID \
    --target-type ip \
    --health-check-path "/health" \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 2 \
    --region $REGION \
    --query 'TargetGroups[0].TargetGroupArn' \
    --output text)

# Create listener
aws elbv2 create-listener \
    --load-balancer-arn $ALB_ARN \
    --protocol HTTP \
    --port $HOST_PORT \
    --default-actions Type=forward,TargetGroupArn=$TARGET_GROUP_ARN \
    --region $REGION

# Save infrastructure information
echo "Saving infrastructure information..."
cat > deploy/infrastructure.env << EOF
export VPC_ID=$VPC_ID
export PUBLIC_SUBNET_ID=$PUBLIC_SUBNET_ID
export PRIVATE_SUBNET_ID=$PRIVATE_SUBNET_ID
export ECS_SG_ID=$ECS_SG_ID
export RDS_SG_ID=$RDS_SG_ID
export CLUSTER_NAME=$CLUSTER_NAME
export SERVICE_NAME=$SERVICE_NAME
export TASK_FAMILY=$TASK_FAMILY
export TASK_EXECUTION_ROLE_ARN=$TASK_EXECUTION_ROLE_ARN
export TASK_ROLE_ARN=$TASK_ROLE_ARN
export DB_INSTANCE_IDENTIFIER=$DB_INSTANCE_IDENTIFIER
export EFS_FILESYSTEM_ID=$EFS_FILESYSTEM_ID
export ALB_ARN=$ALB_ARN
export TARGET_GROUP_ARN=$TARGET_GROUP_ARN
export AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID
export AWS_REGION=$REGION
EOF

echo "Infrastructure setup completed!"
echo "VPC ID: $VPC_ID"
echo "ECS Cluster: $CLUSTER_NAME"
echo "RDS Instance: $DB_INSTANCE_IDENTIFIER"
echo "EFS Filesystem: $EFS_FILESYSTEM_ID"
echo "Load Balancer ARN: $ALB_ARN" 