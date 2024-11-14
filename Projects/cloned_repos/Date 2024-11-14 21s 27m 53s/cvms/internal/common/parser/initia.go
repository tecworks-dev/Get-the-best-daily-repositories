package parser

import (
	"encoding/json"

	"github.com/cosmostation/cvms/internal/common/types"
	"github.com/pkg/errors"
)

func InitiaStakingValidatorParser(resp []byte) ([]types.CosmosStakingValidator, error) {
	var result types.InitiaStakingValidatorsQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, errors.Cause(err)
	}
	commonStakingValidators := make([]types.CosmosStakingValidator, 0)
	for _, validator := range result.Validators {
		commonStakingValidators = append(commonStakingValidators, types.CosmosStakingValidator{
			OperatorAddress: validator.OperatorAddress,
			ConsensusPubkey: validator.ConsensusPubkey,
			Description:     validator.Description,
			Tokens:          "", // initia has multiple tokens on validators, so skip the tokens
		})
	}
	return commonStakingValidators, nil
}
