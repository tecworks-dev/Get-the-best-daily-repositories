package parser

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/cosmostation/cvms/internal/packages/duty/eventnonce/types"
)

// injective
func InjectiveOrchestratorParser(resp []byte) (string, error) {
	var result types.InjectiveOrchestratorQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", fmt.Errorf("parsing error: %s", err.Error())
	}

	return result.OrchestratorAddress, nil
}

func InjectiveEventNonceParser(resp []byte) (float64, error) {
	var result types.InjectiveEventNonceQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}
	eventNonce, err := strconv.ParseFloat(result.LastClaimEvent.EthereumEventNonce, 64)
	if err != nil {
		return 0, fmt.Errorf("converting error: %s", err.Error())
	}
	return eventNonce, nil
}

// gravity-bridge
func GravityBridgeOrchestratorParser(resp []byte) (string, error) {
	var result types.GravityBridgeOrchestratorQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", fmt.Errorf("parsing error: %s", err.Error())
	}

	return result.OrchestratorAddress, nil
}

func GravityBridgeEventNonceParser(resp []byte) (float64, error) {
	var result types.GravityBridgeEventNonceQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}
	eventNonce, err := strconv.ParseFloat(result.EventNonce, 64)
	if err != nil {
		return 0, fmt.Errorf("converting error: %s", err.Error())
	}
	return eventNonce, nil
}

// sommelier
func SommelierOrchestratorParser(resp []byte) (string, error) {
	var result types.SommelierOrchestratorQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", fmt.Errorf("parsing error: %s", err.Error())
	}

	return result.OrchestratorAddress, nil
}

func SommelierEventNonceParser(resp []byte) (float64, error) {
	var result types.GravityBridgeEventNonceQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}
	eventNonce, err := strconv.ParseFloat(result.EventNonce, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}
	return eventNonce, nil
}
