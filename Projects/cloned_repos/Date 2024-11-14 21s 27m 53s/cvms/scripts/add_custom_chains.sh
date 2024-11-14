#!/usr/bin/env bash

YAML_FILE="support_chains.yaml"


# Define the YAML content as a multi-line variable
additional_yaml_content=$(cat <<EOF
cronos:
  protocol_type: cosmos
  # 1cro = 10^18 basecro for Cronos EVM chain
  support_asset:
    denom: basecro
    decimal: 18
  packages:
    - block
    - upgrade
    - uptime

emoney:
  protocol_type: cosmos
  support_asset:
    denom: ungm
    decimal: 6
  packages:
    - block
    - upgrade
    - uptime

onex:
  protocol_type: cosmos
  support_asset:
    denom: aonex
    decimal: 18
  packages:
    - block
    - upgrade
    - uptime
EOF
)

# Append the content to the temporary YAML file
echo "$additional_yaml_content" >> $YAML_FILE
