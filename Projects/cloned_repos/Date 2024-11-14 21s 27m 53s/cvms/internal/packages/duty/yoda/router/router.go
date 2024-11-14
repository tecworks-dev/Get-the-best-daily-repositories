package router

import (
	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/duty/yoda/api"
	"github.com/cosmostation/cvms/internal/packages/duty/yoda/parser"
	"github.com/cosmostation/cvms/internal/packages/duty/yoda/types"
)

func GetStatus(client *common.Exporter, chainName string) (types.CommonYodaStatus, error) {
	var (
		commonYodaQueryPath string
		commonYodaParser    func(resp []byte) (isActive float64, err error)
	)

	switch chainName {
	case "band":
		commonYodaQueryPath = types.BandYodaQueryPath
		commonYodaParser = parser.BandYodaParser

		return api.GetYodaStatus(client, commonYodaQueryPath, commonYodaParser)
	default:
		return types.CommonYodaStatus{}, common.ErrOutOfSwitchCases
	}
}
