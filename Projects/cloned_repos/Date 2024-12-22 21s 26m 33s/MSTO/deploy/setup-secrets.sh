#!/bin/bash
set -e

# Configuration
APP_NAME="msto"
REGION="us-east-1"

echo "Setting up secrets for $APP_NAME..."

# Source infrastructure information
source deploy/infrastructure.env

# Function to prompt for secret value
get_secret_value() {
    local secret_name=$1
    local default_value=$2
    local secret_value

    if [ -n "$default_value" ]; then
        read -p "Enter value for $secret_name (default: $default_value): " secret_value
        secret_value=${secret_value:-$default_value}
    else
        read -p "Enter value for $secret_name: " secret_value
        while [ -z "$secret_value" ]; do
            echo "Value cannot be empty"
            read -p "Enter value for $secret_name: " secret_value
        done
    fi

    echo "$secret_value"
}

# Create secrets in Parameter Store
echo "Creating secrets in Parameter Store..."

# TradingView webhook URL
TRADINGVIEW_WEBHOOK_URL=$(get_secret_value "TradingView Webhook URL")
aws ssm put-parameter \
    --name "/msto/tradingview_webhook_url" \
    --value "$TRADINGVIEW_WEBHOOK_URL" \
    --type SecureString \
    --overwrite \
    --region $REGION

# News API key
NEWS_API_KEY=$(get_secret_value "News API Key")
aws ssm put-parameter \
    --name "/msto/news_api_key" \
    --value "$NEWS_API_KEY" \
    --type SecureString \
    --overwrite \
    --region $REGION

# Strategy parameters
echo "Setting up strategy parameters..."

# Fundamental strategy parameters
MIN_IMPACT_THRESHOLD=$(get_secret_value "Minimum Impact Threshold" "0.3")
aws ssm put-parameter \
    --name "/msto/min_impact_threshold" \
    --value "$MIN_IMPACT_THRESHOLD" \
    --type String \
    --overwrite \
    --region $REGION

MAX_PE_RATIO=$(get_secret_value "Maximum P/E Ratio" "30.0")
aws ssm put-parameter \
    --name "/msto/max_pe_ratio" \
    --value "$MAX_PE_RATIO" \
    --type String \
    --overwrite \
    --region $REGION

MIN_DROP_THRESHOLD=$(get_secret_value "Minimum Drop Threshold" "-5.0")
aws ssm put-parameter \
    --name "/msto/min_drop_threshold" \
    --value "$MIN_DROP_THRESHOLD" \
    --type String \
    --overwrite \
    --region $REGION

# Volatility strategy parameters
MIN_SENTIMENT_THRESHOLD=$(get_secret_value "Minimum Sentiment Threshold" "-0.5")
aws ssm put-parameter \
    --name "/msto/min_sentiment_threshold" \
    --value "$MIN_SENTIMENT_THRESHOLD" \
    --type String \
    --overwrite \
    --region $REGION

BASE_POSITION_SIZE=$(get_secret_value "Base Position Size" "100")
aws ssm put-parameter \
    --name "/msto/base_position_size" \
    --value "$BASE_POSITION_SIZE" \
    --type String \
    --overwrite \
    --region $REGION

# Optional integrations
echo "Setting up optional integrations..."

# Slack webhook (optional)
read -p "Do you want to configure Slack integration? (y/n): " setup_slack
if [[ $setup_slack =~ ^[Yy]$ ]]; then
    SLACK_WEBHOOK_URL=$(get_secret_value "Slack Webhook URL")
    aws ssm put-parameter \
        --name "/msto/slack_webhook_url" \
        --value "$SLACK_WEBHOOK_URL" \
        --type SecureString \
        --overwrite \
        --region $REGION
fi

# Telegram bot (optional)
read -p "Do you want to configure Telegram integration? (y/n): " setup_telegram
if [[ $setup_telegram =~ ^[Yy]$ ]]; then
    TELEGRAM_BOT_TOKEN=$(get_secret_value "Telegram Bot Token")
    aws ssm put-parameter \
        --name "/msto/telegram_bot_token" \
        --value "$TELEGRAM_BOT_TOKEN" \
        --type SecureString \
        --overwrite \
        --region $REGION

    TELEGRAM_CHAT_ID=$(get_secret_value "Telegram Chat ID")
    aws ssm put-parameter \
        --name "/msto/telegram_chat_id" \
        --value "$TELEGRAM_CHAT_ID" \
        --type SecureString \
        --overwrite \
        --region $REGION
fi

# Create a consolidated environment file for local development
echo "Creating local environment file..."
cat > .env.local << EOF
# Environment
ENV=dev
TRADING_MODE=paper

# API Keys
TRADINGVIEW_WEBHOOK_URL=$TRADINGVIEW_WEBHOOK_URL
NEWS_API_KEY=$NEWS_API_KEY

# Strategy Parameters
MIN_IMPACT_THRESHOLD=$MIN_IMPACT_THRESHOLD
MAX_PE_RATIO=$MAX_PE_RATIO
MIN_DROP_THRESHOLD=$MIN_DROP_THRESHOLD
MIN_SENTIMENT_THRESHOLD=$MIN_SENTIMENT_THRESHOLD
BASE_POSITION_SIZE=$BASE_POSITION_SIZE

# Optional Integrations
EOF

if [[ $setup_slack =~ ^[Yy]$ ]]; then
    echo "SLACK_WEBHOOK_URL=$SLACK_WEBHOOK_URL" >> .env.local
fi

if [[ $setup_telegram =~ ^[Yy]$ ]]; then
    echo "TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN" >> .env.local
    echo "TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID" >> .env.local
fi

# Add database configuration from infrastructure setup
echo "
# Database Configuration
DB_CONNECTION_STRING=$(aws ssm get-parameter --name "/msto/db_connection_string" --with-decryption --query 'Parameter.Value' --output text --region $REGION)
" >> .env.local

echo "Secrets setup completed!"
echo "Local environment file created at .env.local"
echo "Remember to add .env.local to .gitignore to prevent committing sensitive information" 