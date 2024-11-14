package parser

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/common/types"
	"github.com/pkg/errors"
)

func CosmosBlockParser(resp []byte) (
	/* block height */ int64,
	/* block timestamp */ time.Time,
	/* block proposer addrss */ string,
	/* txs in the block */ []types.Tx,
	/* last comit block height*/ int64,
	/* block validators signatures */ []types.Signature,
	error,
) {
	var preResult map[string]interface{}
	if err := json.Unmarshal(resp, &preResult); err != nil {
		return 0, time.Time{}, "", nil, 0, nil, err
	}

	_, ok := preResult["jsonrpc"].(string)
	if ok { // v0.34.x
		var resultV34 types.CosmosV34BlockResponse
		if err := json.Unmarshal(resp, &resultV34); err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		heightString, blockTimestamp, lastCommitHeightString := resultV34.Result.Block.Header.Height, resultV34.Result.Block.Header.Time, resultV34.Result.Block.LastCommit.Height

		blockHeight, err := strconv.ParseInt(heightString, 10, 64)
		if err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		lastCommitBlockHeight, err := strconv.ParseInt(lastCommitHeightString, 10, 64)
		if err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		txs := resultV34.Result.Block.Data.Txs
		signatures := resultV34.Result.Block.LastCommit.Signatures
		proposerAddress := resultV34.Result.Block.Header.ProposerAddress
		return blockHeight, blockTimestamp, proposerAddress, txs, lastCommitBlockHeight, signatures, nil
	} else { // tendermint v0.37.x
		var resultV37 types.CosmosV37BlockResponse
		if err := json.Unmarshal(resp, &resultV37); err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		heightString, blockTimestamp, lastCommitHeightString := resultV37.Block.Header.Height, resultV37.Block.Header.Time, resultV37.Block.LastCommit.Height

		blockHeight, err := strconv.ParseInt(heightString, 10, 64)
		if err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		lastCommitBlockHeight, err := strconv.ParseInt(lastCommitHeightString, 10, 64)
		if err != nil {
			return 0, time.Time{}, "", nil, 0, nil, err
		}

		txs := resultV37.Block.Data.Txs
		signatures := resultV37.Block.LastCommit.Signatures
		proposerAddress := resultV37.Block.Header.ProposerAddress
		return blockHeight, blockTimestamp, proposerAddress, txs, lastCommitBlockHeight, signatures, nil
	}
}

func CosmosStatusParser(resp []byte) (
	/* latest block height */ int64,
	/* unexpected error */ error,
) {
	var preResult map[string]interface{}
	if err := json.Unmarshal(resp, &preResult); err != nil {
		return 0, errors.Wrap(err, "failed to unmarshal json in parser")
	}

	_, ok := preResult["jsonrpc"].(string)
	if ok { // v34
		var resultV34 types.CosmosV34StatusResponse
		if err := json.Unmarshal(resp, &resultV34); err != nil {
			return 0, errors.Wrap(err, "failed to unmarshal json in parser")
		}

		blockHeight, err := strconv.ParseInt(resultV34.Result.SyncInfo.LatestBlockHeight, 10, 64)
		if err != nil {
			return 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
		}

		return blockHeight, nil

	} else { // v37
		var resultV37 types.CosmosV37StatusResponse
		if err := json.Unmarshal(resp, &resultV37); err != nil {
			return 0, errors.Wrap(err, "failed to unmarshal json in parser")
		}

		blockHeight, err := strconv.ParseInt(resultV37.SyncInfo.LatestBlockHeight, 10, 64)
		if err != nil {
			return 0, errors.Wrap(err, "failed to convert from stirng to float in parser")
		}

		return blockHeight, nil
	}
}

// TODO: modify this function logic
func CosmosValidatorParser(resp []byte) ([]types.CosmosValidator, int64, error) {
	var validators types.CosmosV34ValidatorResponse
	err := json.Unmarshal(resp, &validators)
	if err != nil {
		return []types.CosmosValidator{}, 0, err
	}

	if len(validators.Result.Validators) == 0 {
		var validators types.CosmosV37ValidatorResponse
		err := json.Unmarshal(resp, &validators)
		if err != nil {
			return []types.CosmosValidator{}, 0, err
		}

		totalValidatorsCount, err := strconv.ParseInt(validators.Total, 10, 64)
		if err != nil {
			return []types.CosmosValidator{}, 0, err
		}

		return validators.Validators, totalValidatorsCount, nil
	} else {
		totalValidatorsCount, err := strconv.ParseInt(validators.Result.Total, 10, 64)
		if err != nil {
			return []types.CosmosValidator{}, 0, err
		}

		return validators.Result.Validators, totalValidatorsCount, nil
	}
}

func CosmosStakingValidatorParser(resp []byte) ([]types.CosmosStakingValidator, error) {
	var result types.CosmosStakingValidatorsQueryResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return nil, common.ErrFailedJsonUnmarshal
	}
	return result.Validators, nil
}
