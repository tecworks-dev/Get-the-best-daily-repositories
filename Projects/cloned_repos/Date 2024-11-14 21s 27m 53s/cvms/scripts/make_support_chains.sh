#!/usr/bin/env bash

# TODO: Check bash version is upper than v4
# echo "Bash version: $BASH_VERSION"


getAssetDetailFromChainlist() {
    chainName="$1"
    jsonURL="https://raw.githubusercontent.com/cosmostation/chainlist/main/chain/${chainName}/assets.json"

    # Fetch the JSON data using curl
    response=$(curl -s "$jsonURL")
    if [ $? -ne 0 ]; then
        echo "Failed to get asset detail" >&2
        return 1
    fi

    denom=$(echo "$response" | jq -r '.[0].denom')
    decimal=$(echo "$response" | jq -r '.[0].decimals')

    if [ -z "$denom" ] || [ -z "$decimal" ]; then
        echo "Failed to parse asset detail for $1, so script will be exited" >&2
        exit 1;
        return 1
    fi

    # Return denom and decimal
    echo "$denom $decimal"
}



# Step 1: Curl the JSON data from the URL
JSON_URL="https://raw.githubusercontent.com/cosmostation/chainlist/main/chain/supports.json"
JSON_DATA=$(curl -s $JSON_URL)

# Check if curl succeeded
if [ $? -ne 0 ]; then
  echo "Failed to fetch JSON data from $JSON_URL"
  exit 1
fi

# Step 2: Extract keys from JSON data
CHAINS=$(echo "$JSON_DATA" | jq -r 'values | .[]')

# Check if jq succeeded
if [ $? -ne 0 ]; then
  echo "Failed to extract keys from JSON"
  exit 1
fi

# List of chains to ignore like non-cosmos chains in supports.json
IGNORE_CHAINS=(
  # evm
  "ethereum" "avalanche" "arbitrum" "optimism" "base" "polygon" "fantom" "moonbeam"
  # move
  "aptos" "sui"
  # bnb
  "bnb-beacon-chain" "bnb-smart-chain"
  # polkadot
  "polkadot"
  # unsupported asset directories
  "okc" "mintstation" "terra-classic"
  # etc...
  "bostrom" "cerberus" "cheqd" "decentr" "dig-chain" "gno" "jupiter" "konstellation" "kujira" "meme" "microtick"
  "odin" "station" "supernova" "tgrade" "vidulum"
)

# List of chains to add supports chain values, these chains are already in chainlist but not in support.json 
EXTERNAL_CHAINS=(
  # add empty space for seperator
  "" 
  # add external chains
  "emoney" "cronos"
)

CHAINS+="${EXTERNAL_CHAINS[@]}"

# Define the packages map (associative array)
declare -A PACKAGE_MAP

# Add eventnonce chains
PACKAGE_MAP["injective"]="eventnonce"
PACKAGE_MAP["gravity-bridge"]="eventnonce"
PACKAGE_MAP["sommelier"]="eventnonce"

# Add oracle chains
PACKAGE_MAP["sei"]="oracle"
PACKAGE_MAP["nibiru"]="oracle"
PACKAGE_MAP["umee"]="oracle"

# Add band-yoda 
PACKAGE_MAP["band"]="yoda"

# Add axelar-evm
PACKAGE_MAP["axelar"]="axelar-evm"


# Step 3: Create the YAML content
TEMP_YAML_FILE="temp_support_chains.yaml"
YAML_FILE="support_chains.yaml"

# Clear the file
echo "---" > $TEMP_YAML_FILE

for chain in $CHAINS; do
  # Skip chains containing 'testnet'
  if [[ "$chain" == *testnet* ]]; then
    continue
  fi

  # Skip chains in the ignore list
  for ignore in "${IGNORE_CHAINS[@]}"; do
    if [[ "$chain" == "$ignore" ]]; then
      continue 2
    fi
  done

  echo "trying to $chain's data from chainlist..."
  
  # chain name key
  echo "$chain:" >> $TEMP_YAML_FILE

  # chain parameters
  echo "  protocol_type: cosmos" >> $TEMP_YAML_FILE
  result=$(getAssetDetailFromChainlist "$chain")
  if [ $? -eq 0 ]; then
      denom=$(echo "$result" | awk '{print $1}')
      decimal=$(echo "$result" | awk '{print $2}')
      echo "  support_asset:" >> $TEMP_YAML_FILE
      echo "    denom: $denom" >> $TEMP_YAML_FILE
      echo "    decimal: $decimal" >> $TEMP_YAML_FILE
  fi

  # support packages
  echo "  packages:" >> $TEMP_YAML_FILE
  echo "    - block" >> $TEMP_YAML_FILE
  echo "    - upgrade" >> $TEMP_YAML_FILE
  echo "    - uptime" >> $TEMP_YAML_FILE
  echo "    - voteindexer" >> $TEMP_YAML_FILE

  # Add specific packages based on the chain name using the map
  if [[ -n "${PACKAGE_MAP[$chain]}" ]]; then
    echo "    - ${PACKAGE_MAP[$chain]}" >> $TEMP_YAML_FILE
  fi

  echo "" >> $TEMP_YAML_FILE  # Add a new line between entries
done

# Check if file was written successfully
if [ $? -eq 0 ]; then
  echo "YAML data successfully written to $TEMP_YAML_FILE"

  echo "Sort the YAML chain name keys alphabetically and save the result to support_chains.yaml"
  yq -P 'sort_keys(.)' "${TEMP_YAML_FILE}" > "${YAML_FILE}"

  echo "Remove temp YARM file" 
  rm ${TEMP_YAML_FILE}
else
  echo "Failed to write YAML data to $TEMP_YAML_FILE"
  exit 1
fi
