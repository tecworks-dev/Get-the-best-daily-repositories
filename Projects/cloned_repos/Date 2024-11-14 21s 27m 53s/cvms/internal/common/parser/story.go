package parser

import (
	"encoding/json"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/common/types"
	sdkhelper "github.com/cosmostation/cvms/internal/helper/sdk"
	"github.com/pkg/errors"
)

func StoryStakingValidatorParser(resp []byte) ([]types.CosmosStakingValidator, error) {
	var result types.StoryStakingValidatorsQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, common.ErrFailedJsonUnmarshal
	}

	stakingValidatorList := make([]types.CosmosStakingValidator, 0)
	for _, validator := range result.Msg.Validators {
		// const Secp256k1 = "/cosmos.crypto.secp256k1.PubKey"
		// const TendermintSecp256k1 = "tendermint/PubKeySecp256k1"
		if validator.ConsensusPubkey.Type != sdkhelper.TendermintSecp256k1 {
			return nil, errors.New("unexpected key type for story validators")
		}

		stakingValidatorList = append(stakingValidatorList, types.CosmosStakingValidator{
			OperatorAddress: validator.OperatorAddress,
			Description:     validator.Description,
			// story not same consensus pubkey result.
			ConsensusPubkey: types.ConsensusPubkey{
				Type: sdkhelper.Secp256k1,
				Key:  validator.ConsensusPubkey.Value,
			},
		})
	}

	return stakingValidatorList, nil
}
