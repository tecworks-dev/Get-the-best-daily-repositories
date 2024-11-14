package parser

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/utility/balance/types"
)

// cosmos balance parser
func CosmosBalanceParser(resp []byte, denom string) (float64, error) {
	var result types.CosmosBalanceResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}

	var stringBalance string
	for _, balance := range result.Balances {
		if balance.Denom == denom {
			stringBalance = balance.Amount
			break
		}
	}

	if stringBalance == "" {
		return 0, fmt.Errorf("failed to get specific denom balnce")
	}

	balance, err := strconv.ParseFloat(stringBalance, 64)
	if err != nil {
		return 0, fmt.Errorf("converting error: %s", err.Error())
	}

	return balance, nil
}

// ethereum balance parser
func EthereumBalanceParser(resp []byte, denom string) (float64, error) {
	var result types.EthereumBalanceResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}

	balance, err := helper.ParsingfromHexaNumberBaseHexaDecimal(helper.HexaNumberToInteger(result.Result))
	if err != nil {
		return 0, fmt.Errorf("converting error: %s", err.Error())
	}

	return float64(balance), nil
}
