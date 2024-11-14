package parser

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/health/block/types"
	"github.com/pkg/errors"
)

// cosmos
func CosmosBlockParser(resp []byte) (float64, float64, error) {
	var preResult map[string]interface{}
	if err := json.Unmarshal(resp, &preResult); err != nil {
		return 0, 0, errors.Wrap(err, "failed to unmarshal json in parser")
	}

	_, ok := preResult["jsonrpc"].(string)
	if ok { // tendermint v0.34.x
		var resultV34 types.CosmosV34BlockResponse
		if err := json.Unmarshal(resp, &resultV34); err != nil {
			return 0, 0, errors.Wrap(err, "failed to unmarshal json in parser")
		}

		timestamp := resultV34.Result.SyncInfo.LatestBlockTime.Unix()
		blockHeight, err := strconv.ParseFloat(resultV34.Result.SyncInfo.LatestBlockHeight, 64)
		if err != nil {
			return 0, 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
		}

		return blockHeight, float64(timestamp), nil

	} else { // tendermint v0.37.x
		var resultV37 types.CosmosV37BlockResponse
		if err := json.Unmarshal(resp, &resultV37); err != nil {
			return 0, 0, fmt.Errorf("parsing error: %s", err.Error())
		}

		timestamp := resultV37.SyncInfo.LatestBlockTime.Unix()
		blockHeight, err := strconv.ParseFloat(resultV37.SyncInfo.LatestBlockHeight, 64)
		if err != nil {
			return 0, 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
		}

		return blockHeight, float64(timestamp), nil
	}
}

// ethereum
func EthereumBlockParser(resp []byte) (float64, float64, error) {
	var result types.EthereumBlockResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, errors.Wrap(err, "failed to unmarshal json in parser")
	}

	timestamp, err := helper.ParsingfromHexaNumberBaseHexaDecimal(helper.HexaNumberToInteger(result.Result.TimeStamp))
	if err != nil {
		return 0, 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
	}

	blockHeight, err := helper.ParsingfromHexaNumberBaseHexaDecimal(helper.HexaNumberToInteger(result.Result.Number))
	if err != nil {
		return 0, 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
	}

	return float64(blockHeight), float64(timestamp), nil
}

// celestia
func CelestiaBlockParser(resp []byte) (float64, float64, error) {
	var result types.CelestiaBlockResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, errors.Wrap(err, "failed to unmarshal json in parser")
	}

	blockHeight, err := strconv.ParseFloat(result.Result.Header.Height, 64)
	if err != nil {
		return 0, 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
	}

	return blockHeight, float64(result.Result.Header.Time.Unix()), nil
}
