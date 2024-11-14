package router

import (
	"github.com/cosmostation/cvms/internal/common"

	"github.com/cosmostation/cvms/internal/packages/consensus/uptime/api"
	"github.com/cosmostation/cvms/internal/packages/consensus/uptime/types"
)

func GetStatus(c *common.Exporter, p common.Packager) (types.CommonUptimeStatus, error) {
	switch p.ProtocolType {
	case "cosmos":
		if p.IsConsumerChain {
			return api.GetConsumserUptimeStatus(c, p.ChainID)
		}
		return api.GetUptimeStatus(c)
	default:
		return types.CommonUptimeStatus{}, common.ErrOutOfSwitchCases
	}
}
