package types

import "time"

var (
	SupportedProtocolTypes = []string{"cosmos"}
)

const (
	// cosmos
	CosmosUpgradeQueryPath     = "/cosmos/upgrade/v1beta1/current_plan"
	CosmosLatestBlockQueryPath = "/cosmos/base/tendermint/v1beta1/blocks/latest"
	CosmosBlockQueryPath       = "/cosmos/base/tendermint/v1beta1/blocks/{height}"

	// celestia
	CelestiaUpgradeQueryPath = "/signal/v1/upgrade"
)

type CommonUpgrade struct {
	RemainingTime float64
	UpgradeName   string
}

// https://lcd-office.cosmostation.io/neutron-testnet/cosmos/upgrade/v1beta1/current_plan
type CosmosUpgradeResponse struct {
	Plan struct {
		Name                string `json:"name"`
		Time                string `json:"time"`
		Height              string `json:"height"`
		Info                string `json:"info"`
		UpgradedClientState string `json:"upgraded_client_state"`
	} `json:"plan"`
}

type CosmosBlockResponse struct {
	BlockId interface{} `json:"-"`
	Block   struct {
		Header struct {
			Height string    `json:"height"`
			Time   time.Time `json:"time"`
		}
		Data       interface{} `json:"-"`
		Evidence   interface{} `json:"-"`
		LastCommit interface{} `json:"-"`
	} `json:"block"`
}

// https://lcd-office.cosmostation.io/celestia-testnet//signal/v1/upgrade
type CelestiaUpgradeResponse struct {
	Upgrade struct {
		AppVersion    string `json:"app_version"`
		UpgradeHeight string `json:"upgrade_height"`
	} `json:"upgrade"`
}
