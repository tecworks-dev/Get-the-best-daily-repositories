package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/duty/eventnonce/api"
	"github.com/cosmostation/cvms/internal/packages/duty/eventnonce/parser"
	"github.com/cosmostation/cvms/internal/packages/duty/eventnonce/types"
)

func GetStatus(c *common.Exporter, chainName string) (types.CommonEventNonceStatus, error) {
	var (
		commonOrchestratorPath   string
		commonOrchestratorParser func(resp []byte) (orchestratorAddress string, err error)

		commonEventNonceQueryPath string
		commonEventNonceParser    func(resp []byte) (eventNonce float64, err error)
	)

	switch chainName {
	case "injective":
		commonOrchestratorPath = types.InjectiveOchestratorQueryPath
		commonOrchestratorParser = parser.InjectiveOrchestratorParser

		commonEventNonceQueryPath = types.InjectiveEventNonceQueryPath
		commonEventNonceParser = parser.InjectiveEventNonceParser

		return api.GetEventNonceStatusByGRPC(c, commonOrchestratorPath, commonOrchestratorParser, commonEventNonceQueryPath, commonEventNonceParser)

	case "gravity-bridge":
		commonOrchestratorPath = types.GravityBridgeOrchestarorQueryPath
		commonOrchestratorParser = parser.GravityBridgeOrchestratorParser

		commonEventNonceQueryPath = types.GravityBridgeEventNonceQueryPath
		commonEventNonceParser = parser.GravityBridgeEventNonceParser

		return api.GetEventNonceStatusByGRPC(c, commonOrchestratorPath, commonOrchestratorParser, commonEventNonceQueryPath, commonEventNonceParser)

	case "sommelier":
		commonOrchestratorPath = types.SommelierOrchestratorQueryPath
		commonOrchestratorParser = parser.SommelierOrchestratorParser

		commonEventNonceQueryPath = types.SommelierEventNonceQueryPath
		commonEventNonceParser = parser.SommelierEventNonceParser

		return api.GetEventNonceStatusByGRPC(c, commonOrchestratorPath, commonOrchestratorParser, commonEventNonceQueryPath, commonEventNonceParser)

	default:
		return types.CommonEventNonceStatus{}, common.ErrOutOfSwitchCases
	}
}
