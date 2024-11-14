package types

const (
	EthereumBlockQueryPath    = ""
	EthereumBlockQueryPayLoad = `{
								"jsonrpc":"2.0",
								"id":1,
								"method":"eth_getBlockByNumber",
								"params":["latest",false]
							}`
)

type EthereumBlockResponse struct {
	JsonRPC string `json:"jsonrpc"`
	ID      int    `json:"id"`
	Result  struct {
		BaseFeePerGas   string   `json:"baseFeePerGas"`
		Difficulty      string   `json:"difficulty"`
		ExtraData       string   `json:"extraData"`
		GasLimit        string   `json:"gasLimit"`
		GasUsed         string   `json:"gasUsed"`
		Hash            string   `json:"hash"`
		LogsBloom       string   `json:"logsBloom"`
		Miner           string   `json:"miner"`
		MinHash         string   `json:"mixHash"`
		Nonce           string   `json:"nonce"`
		Number          string   `json:"number"`
		ParentHash      string   `json:"parentHash"`
		ReceiptsRoot    string   `json:"receiptsRoot"`
		Sha3Uncles      string   `json:"sha3Uncles"`
		Size            string   `json:"size"`
		StateRoot       string   `json:"stateRoot"`
		TimeStamp       string   `json:"timestamp"`
		TotalDifficulty string   `json:"totalDifficulty"`
		Transactions    []string `json:"transactions"`
		TransactionRoot string   `json:"transactionsRoot"`
		Uncles          []string `json:"uncles"`
	} `json:"result"`
}
