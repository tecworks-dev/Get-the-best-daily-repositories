package types

var (
	SupportedChains = []string{"umee", "sei", "nibiru"}
)

const (
	// common api
	CommonValidatorQueryPath   = "/cosmos/staking/v1beta1/validators?status=BOND_STATUS_BONDED&pagination.count_total=true&pagination.limit=500"
	CommonBlockHeightQueryPath = "/cosmos/base/tendermint/v1beta1/blocks/latest"

	// umee paths
	UmeeOracleParamsQueryPath = "/umee/oracle/v1/params"
	UmeeOracleQueryPath       = "/umee/oracle/v1/validators/{validator_address}/miss"

	// kujira paths
	KujiraOracleParamsQueryPath = ""
	KujiraOracleQueryPath       = "/oracle/validators/{validator_address}/miss"

	// sei paths
	SeiOracleParamsQueryPath = "/sei-protocol/sei-chain/oracle/params"
	SeiOracleQueryPath       = "/sei-protocol/sei-chain/oracle/validators/{validator_address}/vote_penalty_counter"

	// nibiru paths
	NibiruOracleParamsQueryPath = "/nibiru/oracle/v1beta1/params"
	NibiruOracleQueryPath       = "/nibiru/oracle/v1beta1/validators/{validator_address}/miss"
)

// common
type CommonOracleStatus struct {
	BlockHeight           float64 `json:"block_height"`
	SlashWindow           float64 `json:"slash_winodw"`
	VotePeriod            float64 `json:"vote_period"`
	MinimumValidPerWindow float64 `json:"min_valid_per_window"`
	VoteWindow            float64 `json:"vote_window"`
	Validators            []ValidatorStatus
}

type ValidatorStatus struct {
	Moniker                  string `json:"moniker"`
	ValidatorOperatorAddress string `json:"validator_operator_address"`
	MissCounter              uint64 `json:"miss_counter"`
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

type CommonLatestBlockQueryResponse struct {
	BlockId interface{} `json:"-"`
	Block   struct {
		Header struct {
			Height string `json:"height"`
		} `json:"header"`
	} `json:"block"`
}

// umee
type UmeeOracleResponse struct {
	MissCounter string `json:"miss_counter"`
}

type UmeeOracleParamsResponse struct {
	Params struct {
		VotePeriod               string      `json:"vote_period"`
		VoteThreshold            float64     `json:"-"`
		RewardBand               string      `json:"-"`
		RewardDistributionWindow string      `json:"-"`
		AcceptList               interface{} `json:"-"`
		SlashFraction            string      `json:"-"`
		SlashWindow              string      `json:"slash_window"`
		MinValidPerWindow        string      `json:"min_valid_per_window"`
		HistoricStampPeriod      string      `json:"-"`
		MedianStampPeriod        string      `json:"-"`
		MaximumPriceStamps       string      `json:"-"`
		MaximumMedianStamps      string      `json:"-"`
	} `json:"params"`
}

// sei
type SeiOracleResponse struct {
	VotePenaltyCounter struct {
		MissCount    string `json:"miss_count"`
		AbstainCount string `json:"abstain_count"`
		SuccessCount string `json:"success_count"`
	} `json:"vote_penalty_counter"`
}

type SeiOracleParamsResponse struct {
	Params struct {
		VotePeriod        string      `json:"vote_period"`
		VoteThreshold     float64     `json:"-"`
		RewardBand        string      `json:"-"`
		Whitelist         interface{} `json:"-"`
		SlashFraction     string      `json:"-"`
		SlashWindow       string      `json:"slash_window"`
		MinValidPerWindow string      `json:"min_valid_per_window"`
		LookBackDuration  string      `json:"-"`
	} `json:"params"`
}

// nibiru
type NibiruOracleResponse struct {
	MissCounter string `json:"miss_counter"`
}

type NibiruOracleParamsResponse struct {
	Params struct {
		VotePeriod        string      `json:"vote_period"`
		VoteThreshold     float64     `json:"-"`
		RewardBand        string      `json:"-"`
		Whitelist         interface{} `json:"-"`
		SlashFraction     string      `json:"-"`
		SlashWindow       string      `json:"slash_window"`
		MinValidPerWindow string      `json:"min_valid_per_window"`
		LookBackDuration  string      `json:"-"`
	} `json:"params"`
}
