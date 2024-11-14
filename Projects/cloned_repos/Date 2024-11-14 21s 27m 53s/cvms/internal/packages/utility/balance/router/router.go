package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/utility/balance/api"
	"github.com/cosmostation/cvms/internal/packages/utility/balance/parser"
	"github.com/cosmostation/cvms/internal/packages/utility/balance/types"
)

func GetCommonHealthChecker() {

}

func GetStatus(client *common.Exporter, p common.Packager) (types.CommonBalance, error) {
	var (
		CommonBalanceCallMethod common.Method
		CommonBalanceQueryPath  string
		CommonBalancePayload    string
		CommonBalanceParser     func(resp []byte, denom string) (float64, error)
	)

	switch p.ProtocolType {
	case "cosmos":
		CommonBalanceCallMethod = common.GET
		CommonBalanceQueryPath = types.CosmosBalanceQueryPath
		CommonBalancePayload = types.CosmosBalancePayload
		CommonBalanceParser = parser.CosmosBalanceParser

		return api.GetBalanceStatus(
			client,
			CommonBalanceCallMethod,
			CommonBalanceQueryPath,
			CommonBalancePayload,
			CommonBalanceParser,
			p.BalanceAddresses, p.BalanceDenom, p.BalanceExponent,
		)

	// NOTE: this is for bridge relayer
	case "ethereum":
		CommonBalanceCallMethod = common.POST
		CommonBalanceQueryPath = types.EthereumBalanceQueryPath
		CommonBalancePayload = types.EthereumBalancePayLoad
		CommonBalanceParser = parser.EthereumBalanceParser

		return api.GetBalanceStatus(
			client,
			CommonBalanceCallMethod,
			CommonBalanceQueryPath,
			CommonBalancePayload,
			CommonBalanceParser,
			p.BalanceAddresses, p.BalanceDenom, p.BalanceExponent,
		)
	default:
		return types.CommonBalance{}, common.ErrOutOfSwitchCases
	}
}
