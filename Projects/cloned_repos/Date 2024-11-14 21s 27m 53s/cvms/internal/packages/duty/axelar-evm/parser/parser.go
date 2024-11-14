package parser

import (
	"encoding/json"

	"github.com/cosmostation/cvms/internal/packages/duty/axelar-evm/types"
)

// axelar
func AxelarEvmChainsParser(resp []byte) ([]string, error) {
	var result types.AxelarEvmChainsResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return []string{}, nil
	}
	return result.Chains, nil
}

func AxelarChainMaintainersParser(resp []byte) ([]string, error) {
	var result types.AxelarChainMaintainersResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return []string{}, nil
	}
	return result.Maintainers, nil
}
