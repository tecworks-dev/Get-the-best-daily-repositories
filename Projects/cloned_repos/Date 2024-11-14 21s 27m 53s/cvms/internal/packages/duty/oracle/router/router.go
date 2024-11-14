package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/api"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/parser"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/types"
)

func GetStatus(client *common.Exporter, chainName string) (types.CommonOracleStatus, error) {
	var (
		commonOracleQueryPath       string
		commonOracleParser          func(resp []byte) (missCount uint64, err error)
		commonOracleParamsQueryPath string
		commonOracleParamsParser    func(resp []byte) (slashWindow, votePeriod, minValidPerWindow, voteWindow float64, err error)
	)

	switch chainName {
	case "umee":
		commonOracleQueryPath = types.UmeeOracleQueryPath
		commonOracleParser = parser.UmeeOracleParser

		commonOracleParamsQueryPath = types.UmeeOracleParamsQueryPath
		commonOracleParamsParser = parser.UmeeOracleParamParser

		return api.GetOracleStatus(client, commonOracleQueryPath, commonOracleParser, commonOracleParamsQueryPath, commonOracleParamsParser)

	case "sei":
		commonOracleQueryPath = types.SeiOracleQueryPath
		commonOracleParser = parser.SeiOracleParser

		commonOracleParamsQueryPath = types.SeiOracleParamsQueryPath
		commonOracleParamsParser = parser.SeiOracleParamParser

		return api.GetOracleStatus(client, commonOracleQueryPath, commonOracleParser, commonOracleParamsQueryPath, commonOracleParamsParser)

	case "nibiru":
		commonOracleQueryPath = types.NibiruOracleQueryPath
		commonOracleParser = parser.NibiruOracleParser

		commonOracleParamsQueryPath = types.NibiruOracleParamsQueryPath
		commonOracleParamsParser = parser.NibiruOracleParamParser

		return api.GetOracleStatus(client, commonOracleQueryPath, commonOracleParser, commonOracleParamsQueryPath, commonOracleParamsParser)

	default:
		return types.CommonOracleStatus{}, common.ErrOutOfSwitchCases
	}
}
