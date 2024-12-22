#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"
DB_INSTANCE_IDENTIFIER="$APP_NAME-db"
DB_NAME="msto"
DB_USERNAME="msto_user"
DB_PORT=5432

echo "Setting up RDS database for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Generate random password
DB_PASSWORD=$(openssl rand -base64 32)

# Create DB subnet group
echo "Creating DB subnet group..."
aws rds create-db-subnet-group \
    --db-subnet-group-name "$APP_NAME-subnet-group" \
    --db-subnet-group-description "Subnet group for MSTO database" \
    --subnet-ids $SUBNET_IDS \
    --region $REGION

# Create DB security group
echo "Creating DB security group..."
DB_SG_ID=$(aws ec2 create-security-group \
    --group-name "$APP_NAME-db-sg" \
    --description "Security group for MSTO database" \
    --vpc-id $VPC_ID \
    --query "GroupId" \
    --output text)

# Allow PostgreSQL access from ECS security group
aws ec2 authorize-security-group-ingress \
    --group-id $DB_SG_ID \
    --protocol tcp \
    --port $DB_PORT \
    --source-group $ECS_SG_ID

# Create RDS instance
echo "Creating RDS instance..."
aws rds create-db-instance \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --db-name $DB_NAME \
    --engine postgres \
    --engine-version 14.7 \
    --db-instance-class db.t3.micro \
    --allocated-storage 20 \
    --storage-type gp2 \
    --master-username $DB_USERNAME \
    --master-user-password "$DB_PASSWORD" \
    --vpc-security-group-ids $DB_SG_ID \
    --db-subnet-group-name "$APP_NAME-subnet-group" \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00" \
    --preferred-maintenance-window "Mon:04:00-Mon:05:00" \
    --multi-az false \
    --publicly-accessible false \
    --enable-performance-insights \
    --performance-insights-retention-period 7 \
    --enable-cloudwatch-logs-exports '["postgresql","upgrade"]' \
    --deletion-protection \
    --tags Key=Application,Value=$APP_NAME \
    --region $REGION

# Wait for DB instance to be available
echo "Waiting for RDS instance to be available..."
aws rds wait db-instance-available \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --region $REGION

# Get DB endpoint
DB_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier $DB_INSTANCE_IDENTIFIER \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text \
    --region $REGION)

# Store database credentials in SSM Parameter Store
echo "Storing database credentials in SSM Parameter Store..."
aws ssm put-parameter \
    --name "/msto/db_connection_string" \
    --value "postgresql://$DB_USERNAME:$DB_PASSWORD@$DB_ENDPOINT:$DB_PORT/$DB_NAME" \
    --type SecureString \
    --overwrite \
    --region $REGION

aws ssm put-parameter \
    --name "/msto/db_host" \
    --value "$DB_ENDPOINT" \
    --type String \
    --overwrite \
    --region $REGION

aws ssm put-parameter \
    --name "/msto/db_name" \
    --value "$DB_NAME" \
    --type String \
    --overwrite \
    --region $REGION

aws ssm put-parameter \
    --name "/msto/db_user" \
    --value "$DB_USERNAME" \
    --type String \
    --overwrite \
    --region $REGION

aws ssm put-parameter \
    --name "/msto/db_password" \
    --value "$DB_PASSWORD" \
    --type SecureString \
    --overwrite \
    --region $REGION

# Create database schema
echo "Creating database schema..."
cat > deploy/schema.sql << EOF
-- Create signals table
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    strategy VARCHAR(50) NOT NULL,
    action VARCHAR(10) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2),
    impact DECIMAL(5,2),
    sentiment DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    executed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending',
    error_message TEXT
);

-- Create events table
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_date DATE NOT NULL,
    description TEXT,
    impact DECIMAL(5,2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create metrics table
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,2) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_signals_ticker ON signals(ticker);
CREATE INDEX idx_signals_strategy ON signals(strategy);
CREATE INDEX idx_signals_created_at ON signals(created_at);
CREATE INDEX idx_events_ticker ON events(ticker);
CREATE INDEX idx_events_event_date ON events(event_date);
CREATE INDEX idx_metrics_ticker ON metrics(ticker);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);

-- Create views
CREATE OR REPLACE VIEW signal_summary AS
SELECT
    ticker,
    strategy,
    action,
    COUNT(*) as signal_count,
    AVG(impact) as avg_impact,
    AVG(sentiment) as avg_sentiment,
    MIN(created_at) as first_signal,
    MAX(created_at) as last_signal
FROM signals
GROUP BY ticker, strategy, action;

CREATE OR REPLACE VIEW event_summary AS
SELECT
    ticker,
    event_type,
    COUNT(*) as event_count,
    AVG(impact) as avg_impact,
    MIN(event_date) as first_event,
    MAX(event_date) as last_event
FROM events
GROUP BY ticker, event_type;
EOF

# Apply schema
echo "Applying database schema..."
PGPASSWORD="$DB_PASSWORD" psql \
    -h $DB_ENDPOINT \
    -U $DB_USERNAME \
    -d $DB_NAME \
    -f deploy/schema.sql

echo "Database setup completed!"
echo "Connection string stored in SSM Parameter Store: /msto/db_connection_string" 