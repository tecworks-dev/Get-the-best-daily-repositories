package types

import "fmt"

type InitiaStakingValidatorsQueryResponse struct {
	Validators []InitiaStakingValidator `json:"validators"`
	Pagination struct {
		// NextKey interface{} `json:"-"`
		Total string `json:"total"`
	} `json:"pagination"`
}

type InitiaStakingValidator struct {
	OperatorAddress string          `json:"operator_address"`
	ConsensusPubkey ConsensusPubkey `json:"consensus_pubkey"`
	Description     struct {
		Moniker string `json:"moniker"`
	} `json:"description"`
	Tokens interface{} `json:"-"`
}

// ref; https://github.com/initia-labs/initia/blob/main/proto/initia/mstaking/v1/query.proto#L14
var InitiaStakingValidatorQueryPath = func(status string) string {
	return fmt.Sprintf("/initia/mstaking/v1/validators?status=%s&pagination.count_total=true&pagination.limit=500", status)
}
