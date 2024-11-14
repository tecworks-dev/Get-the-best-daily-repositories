package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/api"
	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/parser"
	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/types"
)

func GetStatus(client *common.Exporter, chainName string) (types.CommonUpgrade, error) {
	var (
		commonUpgradeQueryPath string
		commonUpgradeParser    func([]byte) (int64, string, error)

		commonBlockQueryPath string
		commonBlockParser    func([]byte) (int64, int64, error)

		commonLatestBlockQueryPath string
	)

	switch chainName {
	case "celestia":
		commonUpgradeQueryPath = types.CelestiaUpgradeQueryPath
		commonUpgradeParser = parser.CelestiaUpgradeParser

		commonLatestBlockQueryPath = types.CosmosLatestBlockQueryPath
		commonBlockQueryPath = types.CosmosBlockQueryPath
		commonBlockParser = parser.CosmosBlockParser

		return api.GetUpgradeStatus(client,
			commonUpgradeQueryPath, commonUpgradeParser,
			commonBlockQueryPath, commonBlockParser,
			commonLatestBlockQueryPath,
		)

	default:
		commonUpgradeQueryPath = types.CosmosUpgradeQueryPath
		commonUpgradeParser = parser.CosmosUpgradeParser

		commonLatestBlockQueryPath = types.CosmosLatestBlockQueryPath
		commonBlockQueryPath = types.CosmosBlockQueryPath
		commonBlockParser = parser.CosmosBlockParser

		return api.GetUpgradeStatus(client,
			commonUpgradeQueryPath, commonUpgradeParser,
			commonBlockQueryPath, commonBlockParser,
			commonLatestBlockQueryPath,
		)
	}
}
