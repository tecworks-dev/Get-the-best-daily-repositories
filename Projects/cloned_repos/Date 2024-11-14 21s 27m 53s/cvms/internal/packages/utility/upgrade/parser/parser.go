package parser

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/types"
)

// cosmos upgrade parser
func CosmosUpgradeParser(resp []byte) (
	/* upgrade height */ int64,
	/* upgrade plan name  */ string,
	error) {
	var result types.CosmosUpgradeResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, "", fmt.Errorf("parsing error: %s", err.Error())
	}

	if result.Plan.Height == "" {
		return 0, "", nil
	}

	upgradeHeight, err := strconv.ParseInt(result.Plan.Height, 10, 64)
	if err != nil {
		return 0, "", fmt.Errorf("converting error: %s", err.Error())
	}
	return upgradeHeight, result.Plan.Name, nil
}

// celestia upgrade parser
func CelestiaUpgradeParser(resp []byte) (
	/* upgrade height */ int64,
	/* upgrade plan name  */ string,
	error) {
	var result types.CelestiaUpgradeResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, "", fmt.Errorf("parsing error: %s", err.Error())
	}

	if result.Upgrade.UpgradeHeight == "" {
		return 0, "", nil
	}

	upgradeHeight, err := strconv.ParseInt(result.Upgrade.UpgradeHeight, 10, 64)
	if err != nil {
		return 0, "", fmt.Errorf("converting error: %s", err.Error())
	}
	return upgradeHeight, result.Upgrade.AppVersion, nil
}

// cosmos latest block parser
func CosmosBlockParser(resp []byte) (
	/* block height */ int64,
	/* block timestamp */ int64,
	error) {
	var result types.CosmosBlockResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, fmt.Errorf("parsing error: %s", err.Error())
	}

	blockTimestamp := result.Block.Header.Time.Unix()
	blockHeight, err := strconv.ParseInt(result.Block.Header.Height, 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("converting error: %s", err.Error())
	}

	return blockHeight, blockTimestamp, nil
}
