package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/duty/axelar-evm/api"
	"github.com/cosmostation/cvms/internal/packages/duty/axelar-evm/parser"
	"github.com/cosmostation/cvms/internal/packages/duty/axelar-evm/types"
)

func GetStatus(exporter *common.Exporter, chainName string) (types.CommonAxelarNexus, error) {
	var (
		commonEvmChainsQueryPath string
		commonEvmChainsParser    func(resp []byte) (activatedEvmChains []string, err error)

		commonEvmChainMaintainerQueryPath string
		commonChainMaintainersParser      func(resp []byte) ([]string, error)
	)

	switch chainName {
	case "axelar":
		commonEvmChainsQueryPath = types.AxelarEvmChainsQueryPath
		commonEvmChainsParser = parser.AxelarEvmChainsParser

		commonEvmChainMaintainerQueryPath = types.AxelarChainMaintainersQueryPath
		commonChainMaintainersParser = parser.AxelarChainMaintainersParser

		return api.GetAxelarNexusStatus(
			exporter,
			commonEvmChainsQueryPath, commonEvmChainsParser,
			commonEvmChainMaintainerQueryPath, commonChainMaintainersParser,
		)

	default:
		return types.CommonAxelarNexus{}, common.ErrOutOfSwitchCases
	}
}
