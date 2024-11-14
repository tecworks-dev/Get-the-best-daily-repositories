package types

import "time"

const (
	CosmosBlockQueryPath    = "/status"
	CosmosBlockQueryPayload = ""
)

type CosmosV34BlockResponse struct {
	JsonRPC string `json:"jsonrpc" validate:"required"`
	ID      int    `json:"id" validate:"required"`
	Result  struct {
		NodeInfo map[string]any `json:"node_info"`
		SyncInfo struct {
			LatestBlockHash     string    `json:"latest_block_hash"`
			LatestAppHash       string    `json:"latest_app_hash"`
			LatestBlockHeight   string    `json:"latest_block_height"`
			LatestBlockTime     time.Time `json:"latest_block_time"`
			EarliestBlockHash   string    `json:"earliest_block_hash"`
			EarliestAppHash     string    `json:"earliest_app_hash"`
			EarliestBlcokHeight string    `json:"earliest_block_height"`
			EarliestBlockTime   time.Time `json:"earliest_block_time"`
			CatchingUp          bool      `json:"catching_up"`
		} `json:"sync_info"`
		ValidatorInfo map[string]any `json:"validator_info"`
	} `json:"result" validate:"required"`
}

type CosmosV37BlockResponse struct {
	NodeInfo map[string]any `json:"node_info"`
	SyncInfo struct {
		LatestBlockHash     string    `json:"latest_block_hash"`
		LatestAppHash       string    `json:"latest_app_hash"`
		LatestBlockHeight   string    `json:"latest_block_height"`
		LatestBlockTime     time.Time `json:"latest_block_time"`
		EarliestBlockHash   string    `json:"earliest_block_hash"`
		EarliestAppHash     string    `json:"earliest_app_hash"`
		EarliestBlcokHeight string    `json:"earliest_block_height"`
		EarliestBlockTime   time.Time `json:"earliest_block_time"`
		CatchingUp          bool      `json:"catching_up"`
	} `json:"sync_info" validate:"required"`
	ValidatorInfo map[string]any `json:"validator_info"`
}
