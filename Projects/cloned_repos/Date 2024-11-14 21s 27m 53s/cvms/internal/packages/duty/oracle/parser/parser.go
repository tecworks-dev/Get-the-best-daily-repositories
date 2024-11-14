package parser

import (
	"encoding/json"
	"strconv"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/types"
)

// umee
func UmeeOracleParser(resp []byte) (uint64, error) {
	var result types.UmeeOracleResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, err
	}

	misscount, err := helper.ParsingfromHexaNumberBaseDecimal(result.MissCounter)
	if err != nil {
		return 0, err
	}

	return misscount, nil
}

func UmeeOracleParamParser(resp []byte) (float64, float64, float64, float64, error) {
	var result types.UmeeOracleParamsResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, 0, 0, err
	}

	slashWindow, err := strconv.ParseFloat(result.Params.SlashWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	votePeriod, err := strconv.ParseFloat(result.Params.VotePeriod, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	minValidPerWindow, err := strconv.ParseFloat(result.Params.MinValidPerWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	voteWindow := (slashWindow / votePeriod)
	return slashWindow, votePeriod, minValidPerWindow, voteWindow, nil
}

// sei
func SeiOracleParser(resp []byte) (uint64, error) {
	var result types.SeiOracleResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, err
	}

	missCount, err := helper.ParsingfromHexaNumberBaseDecimal(result.VotePenaltyCounter.MissCount)
	if err != nil {
		return 0, err
	}

	abstainCount, err := helper.ParsingfromHexaNumberBaseDecimal(result.VotePenaltyCounter.AbstainCount)
	if err != nil {
		return 0, err
	}

	return missCount + abstainCount, nil
}

func SeiOracleParamParser(resp []byte) (float64, float64, float64, float64, error) {
	var result types.SeiOracleParamsResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, 0, 0, err
	}

	slashWindow, err := strconv.ParseFloat(result.Params.SlashWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	votePeriod, err := strconv.ParseFloat(result.Params.VotePeriod, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	minValidPerWindow, err := strconv.ParseFloat(result.Params.MinValidPerWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	voteWindow := (slashWindow / votePeriod)

	return slashWindow, votePeriod, minValidPerWindow, voteWindow, nil
}

// nibiru
func NibiruOracleParser(resp []byte) (uint64, error) {
	var result types.NibiruOracleResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, err
	}

	misscount, err := helper.ParsingfromHexaNumberBaseDecimal(result.MissCounter)
	if err != nil {
		return 0, err
	}

	return misscount, nil
}

func NibiruOracleParamParser(resp []byte) (float64, float64, float64, float64, error) {
	var result types.NibiruOracleParamsResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, 0, 0, err
	}

	slashWindow, err := strconv.ParseFloat(result.Params.SlashWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	votePeriod, err := strconv.ParseFloat(result.Params.VotePeriod, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	minValidPerWindow, err := strconv.ParseFloat(result.Params.MinValidPerWindow, 64)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	voteWindow := (slashWindow / votePeriod)

	return slashWindow, votePeriod, minValidPerWindow, voteWindow, nil
}
