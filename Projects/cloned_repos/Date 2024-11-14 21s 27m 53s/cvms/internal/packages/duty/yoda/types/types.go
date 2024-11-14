package types

import "time"

var (
	SupportedChains = []string{"band"}
)

const (
	// common
	CommonValidatorQueryPath = "/cosmos/staking/v1beta1/validators?status=BOND_STATUS_BONDED&pagination.count_total=true&pagination.limit=500"

	// band paths
	BandYodaQueryPath = "/oracle/v1/validators/{validator_address}"
)

// common
type CommonYodaStatus struct {
	Validators []ValidatorStatus
}

type ValidatorStatus struct {
	Moniker                  string  `json:"moniker"`
	ValidatorOperatorAddress string  `json:"validator_operator_address"`
	IsActive                 float64 `json:"is_active"`
}

type CommonValidatorsQueryResponse struct {
	Validators []struct {
		OperatorAddress string `json:"operator_address"`
		Description     struct {
			Moniker string `json:"moniker"`
		} `json:"description"`
	} `json:"validators"`
	Pagination struct {
		NextKey interface{} `json:"-"`
		Total   string      `json:"-"`
	} `json:"-"`
}

// band
type BandYodaResponse struct {
	Status struct {
		IsActive bool      `json:"is_active"`
		Since    time.Time `json:"since"`
	} `json:"status"`
}
