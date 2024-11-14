package types

var (
	// common
	SupportedProtocolTypes = []string{"cosmos", "ethereum"}
)

const (
	// cosmos
	CosmosBalanceQueryPath = "/cosmos/bank/v1beta1/balances/{balance_address}"
	CosmosBalancePayload   = ""

	// ethereum
	EthereumBalanceQueryPath = ""
	EthereumBalancePayLoad   = `{
								"jsonrpc":"2.0",
								"id":1,
								"method":"eth_getBalance",
								"params":["{balance_address}","latest"]
							}`
)

type CommonBalance struct {
	Balances []BalanceStatus
}

type BalanceStatus struct {
	Address          string
	RemainingBalance float64
}

type CosmosBalanceResponse struct {
	Balances []struct {
		Denom  string `json:"denom"`
		Amount string `json:"amount"`
	} `json:"balances"`
	Pagination interface{} `json:"-"`
}

type EthereumBalanceResponse struct {
	JsonRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Result  string `json:"result"`
}
