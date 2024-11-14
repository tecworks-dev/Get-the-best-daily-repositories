package parser

import (
	"encoding/json"
	"fmt"

	"github.com/cosmostation/cvms/internal/packages/duty/yoda/types"
)

// band
func BandYodaParser(resp []byte) (float64, error) {
	var result types.BandYodaResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, fmt.Errorf("parsing error: %s", err.Error())
	}
	if !result.Status.IsActive {
		return 0, nil
	}
	return 1, nil
}
