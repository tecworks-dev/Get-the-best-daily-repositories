package function

import (
	"encoding/base64"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/common/api"
	indexermodel "github.com/cosmostation/cvms/internal/common/indexer/model"
	"github.com/cosmostation/cvms/internal/common/types"

	sdkhelper "github.com/cosmostation/cvms/internal/helper/sdk"
	"github.com/pkg/errors"
)

func MakeValidatorInfoList(
	app common.CommonApp,
	chainID string, chainInfoID int64,
	chainName string, isConsumer bool,
	newValidatorAddressMap map[string]bool,
) ([]indexermodel.ValidatorInfo, error) {
	switch true {
	case chainName == "bera":
		// berachain doens't have staking module, so we need to make custom validator info list for bera
		// stakingValidators, err := api.GetValidators(commonClient, chainName)
		newStakingValidatorMap := make(map[string]types.StakingValidatorMetaInfo)
		validators, err := api.GetValidators(app.CommonClient)
		if err != nil {
			return nil, errors.Cause(err)
		}
		// TODO: change logic with staking module contract on bera-geth
		for _, validator := range validators {
			hexAddress := validator.Address
			blsPubkey, _ := sdkhelper.MakeBLSPubkey(validator.Pubkey.Value)
			newStakingValidatorMap[hexAddress] = types.StakingValidatorMetaInfo{
				Moniker:         blsPubkey,
				OperatorAddress: blsPubkey,
			}
		}

		newValidatorInfoList := make([]indexermodel.ValidatorInfo, 0)
		for newHexAddress := range newValidatorAddressMap {
			if _, exist := newStakingValidatorMap[newHexAddress]; !exist {
				return nil, errors.New("unexpected error while collecting validator info data")
			}
			newValidatorInfoList = append(
				newValidatorInfoList,
				indexermodel.ValidatorInfo{
					ChainInfoID:     chainInfoID,
					HexAddress:      newHexAddress,
					OperatorAddress: newStakingValidatorMap[newHexAddress].OperatorAddress,
					Moniker:         newStakingValidatorMap[newHexAddress].Moniker,
				})
		}
		return newValidatorInfoList, nil
	case isConsumer:
		providerClient := app.OptionalClient

		var consumerID string
		consumerChains, err := api.GetConsumerChainID(providerClient)
		if err != nil {
			return nil, errors.Cause(err)
		}
		for _, consumerChain := range consumerChains {
			if consumerChain.ChainID == chainID {
				consumerID = consumerChain.ConsumerID
				break
			}
		}
		if consumerID == "" {
			return nil, errors.Errorf("failed to find consumer id, check again your chain-id: %s", chainID)
		}

		newStakingValidatorMap := make(map[string]types.StakingValidatorMetaInfo)
		stakingProviderValidators, err := api.GetProviderValidators(providerClient, consumerID)
		if err != nil {
			return nil, errors.Cause(err)
		}
		for _, validator := range stakingProviderValidators {
			decodedKey, _ := base64.StdEncoding.DecodeString(validator.ConsumerKey.Pubkey)
			hexAddress, err := sdkhelper.MakeProposerAddress(sdkhelper.Ed25519, decodedKey)
			if err != nil {
				return nil, errors.Cause(err)
			}
			newStakingValidatorMap[hexAddress] = types.StakingValidatorMetaInfo{
				Moniker:         validator.Description.Moniker,
				OperatorAddress: validator.ProviderValoperAddress,
			}
		}
		retryCount := 0
		newValidatorInfoList := make([]indexermodel.ValidatorInfo, 0)
		for newHexAddress := range newValidatorAddressMap {
		consumerRetryLoop:
			if _, exist := newStakingValidatorMap[newHexAddress]; !exist {
				app.Warnln("some validators aren't in BOND_STATUS_BONDED, retry after getting validator info from not BOND_STATUS", "Retrying... Attempt", retryCount)
				retryCount++
				switch retryCount {
				case 1:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unbonding)
					goto consumerRetryLoop
				case 2:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unbonded)
					goto consumerRetryLoop
				case 3:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unspecfied)
					goto consumerRetryLoop
				default:
					return nil, errors.New("unexpected error while collecting validator info data")
				}
			}
			newValidatorInfoList = append(
				newValidatorInfoList,
				indexermodel.ValidatorInfo{
					ChainInfoID:     chainInfoID,
					HexAddress:      newHexAddress,
					OperatorAddress: newStakingValidatorMap[newHexAddress].OperatorAddress,
					Moniker:         newStakingValidatorMap[newHexAddress].Moniker,
				})
		}
		return newValidatorInfoList, nil
	default:
		// if there is a new validators, first make stakingvalidator map by hexAddress
		newStakingValidatorMap := make(map[string]types.StakingValidatorMetaInfo)
		stakingValidators, err := api.GetStakingValidators(app.CommonClient, chainName)
		if err != nil {
			return nil, errors.Cause(err)
		}
		for _, validator := range stakingValidators {
			decodedKey, _ := base64.StdEncoding.DecodeString(validator.ConsensusPubkey.Key)
			hexAddress, err := sdkhelper.MakeProposerAddress(validator.ConsensusPubkey.Type, decodedKey)
			if err != nil {
				return nil, errors.Cause(err)
			}
			newStakingValidatorMap[hexAddress] = types.StakingValidatorMetaInfo{
				Moniker:         validator.Description.Moniker,
				OperatorAddress: validator.OperatorAddress,
			}
		}
		retryCount := 0
		newValidatorInfoList := make([]indexermodel.ValidatorInfo, 0)
		for newHexAddress := range newValidatorAddressMap {
		retryLoop:
			if _, exist := newStakingValidatorMap[newHexAddress]; !exist {
				app.Warnln("some validators aren't in BOND_STATUS_BONDED, retry after getting validator info from not BOND_STATUS", "Retrying... Attempt", retryCount)
				retryCount++
				switch retryCount {
				case 1:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unbonding)
					goto retryLoop
				case 2:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unbonded)
					goto retryLoop
				case 3:
					GetStakingValidators(app.CommonClient, chainName, newStakingValidatorMap, types.Unspecfied)
					goto retryLoop
				default:
					return nil, errors.New("unexpected error while collecting validator info data")
				}
			}
			newValidatorInfoList = append(
				newValidatorInfoList,
				indexermodel.ValidatorInfo{
					ChainInfoID:     chainInfoID,
					HexAddress:      newHexAddress,
					OperatorAddress: newStakingValidatorMap[newHexAddress].OperatorAddress,
					Moniker:         newStakingValidatorMap[newHexAddress].Moniker,
				})
		}
		return newValidatorInfoList, nil
	}
}

func GetStakingValidators(client common.CommonClient, chainName string, newStakingValidatorMap map[string]types.StakingValidatorMetaInfo, status types.BondStatus) error {
	stakingValidators, err := api.GetStakingValidators(client, chainName, string(status))
	if err != nil {
		return errors.Cause(err)
	}
	for _, validator := range stakingValidators {
		decodedKey, _ := base64.StdEncoding.DecodeString(validator.ConsensusPubkey.Key)
		hexAddress, err := sdkhelper.MakeProposerAddress(validator.ConsensusPubkey.Type, decodedKey)
		if err != nil {
			return errors.Cause(err)
		}
		newStakingValidatorMap[hexAddress] = types.StakingValidatorMetaInfo{
			Moniker:         validator.Description.Moniker,
			OperatorAddress: validator.OperatorAddress,
		}
	}
	return nil
}
