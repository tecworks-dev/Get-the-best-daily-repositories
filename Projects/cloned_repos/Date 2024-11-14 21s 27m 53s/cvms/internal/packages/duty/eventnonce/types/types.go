package types

var (
	SupportedChains = []string{"injective", "gravity-bridge", "sommelier"}
)

const (
	// common api
	CommonValidatorQueryPath = "/cosmos/staking/v1beta1/validators?status=BOND_STATUS_BONDED&pagination.count_total=true&pagination.limit=500"

	// common grpc
	CommonValidatorGrpcQueryPath   = "cosmos.staking.v1beta1.Query.Validators"
	CommonValidatorGrpcQueryOption = `{"status":"BOND_STATUS_BONDED"}`

	// injective paths
	// InjectiveOchestratorQueryPath = "/peggy/v1/query_delegate_keys_by_validator"
	// InjectiveEventNonceQueryPath  = "/peggy/v1/oracle/event/{orchestrator_address}"
	InjectiveOchestratorQueryPath = "injective.peggy.v1.Query.GetDelegateKeyByValidator"
	InjectiveEventNonceQueryPath  = "injective.peggy.v1.Query.LastEventByAddr"

	// gravity-bridge paths
	// GravityBridgeOrchestarorQueryPath = "/gravity/v1beta/query_delegate_keys_by_validator"
	// GravityBridgeEventNonceQueryPath  = "/gravity/v1beta/oracle/eventnonce/{orchestrator_address}"
	GravityBridgeOrchestarorQueryPath = "gravity.v1.Query.GetDelegateKeyByValidator"
	GravityBridgeEventNonceQueryPath  = "gravity.v1.Query.LastEventNonceByAddr"

	// sommelier paths
	SommelierOrchestratorQueryPath = "gravity.v1.Query.DelegateKeysByValidator"
	SommelierEventNonceQueryPath   = "gravity.v1.Query.LastSubmittedEthereumEvent"
)

// common
type CommonEventNonceStatus struct {
	HeighestNonce float64 `json:"heighest_nonce"`
	Validators    []ValidatorStatus
}

type ValidatorStatus struct {
	Moniker                  string  `json:"moniker"`
	ValidatorOperatorAddress string  `json:"validator_operator_address"`
	OrchestratorAddress      string  `json:"orchestrator_addreess"`
	EventNonce               float64 `json:"event_nonce"`
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

// umee
type UmeeOrchestratorQueryResponse struct {
	EthAddress          string `json:"eth_address"`
	OrchestratorAddress string `json:"orchestrator_address"`
}

type UmeeEventNonceQueryResponse struct {
	EventNonce string `json:"event_nonce"`
}

// injective
type InjectiveOrchestratorQueryResponse struct {
	EthAddress          string `json:"eth_address"`
	OrchestratorAddress string `json:"orchestrator_address"`
}

type InjectiveEventNonceQueryResponse struct {
	LastClaimEvent struct {
		EthereumEventNonce  string `json:"ethereum_event_nonce"`
		EthereumEventHeight string `json:"ethereum_event_height"`
	} `json:"last_claim_event"`
}

// onomy
type OnomyOrchestratorQueryResponse struct {
	EthAddress          string `json:"eth_address"`
	OrchestratorAddress string `json:"orchestrator_address"`
}

type OnomyEventNonceQueryResponse struct {
	EventNonce string `json:"event_nonce"`
}

// gravity-bridge
type GravityBridgeOrchestratorQueryResponse struct {
	EthAddress          string `json:"eth_address"`
	OrchestratorAddress string `json:"orchestrator_address"`
}

type GravityBridgeEventNonceQueryResponse struct {
	EventNonce string `json:"event_nonce"`
}

// sommelier
type SommelierOrchestratorQueryResponse struct {
	EthAddress          string `json:"eth_address"`
	OrchestratorAddress string `json:"orchestrator_address"`
}
