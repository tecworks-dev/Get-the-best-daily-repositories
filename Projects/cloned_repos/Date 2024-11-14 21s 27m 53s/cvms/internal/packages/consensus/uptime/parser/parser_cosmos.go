package parser

import (
	"encoding/json"
	"strconv"

	commontypes "github.com/cosmostation/cvms/internal/common/types"
)

func CosmosUptimeParser(resp []byte) (consensusAddress string, indexOffset float64, isTomstoned float64, missedBlocksCounter float64, err error) {
	var result commontypes.CosmosSlashingResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return "", 0, 0, 0, err
	}
	indexOffset, err = strconv.ParseFloat(result.ValidatorSigningInfo.IndexOffset, 64)
	if err != nil {
		return "", 0, 0, 0, err
	}
	missedBlocksCounter, err = strconv.ParseFloat(result.ValidatorSigningInfo.MissedBlocksCounter, 64)
	if err != nil {
		return "", 0, 0, 0, err
	}

	isTomstoned = float64(0)
	if result.ValidatorSigningInfo.Tombstoned {
		isTomstoned = 1
	}

	return result.ValidatorSigningInfo.ConsensusAddress, indexOffset, isTomstoned, missedBlocksCounter, nil
}

func CosmosUptimeParamsParser(resp []byte) (signedBlocksWindow float64, minSignedPerWindow float64, err error) {
	var result commontypes.CosmosSlashingParamsResponse
	if err := json.Unmarshal(resp, &result); err != nil {
		return 0, 0, err
	}
	signedBlocksWindow, err = strconv.ParseFloat(result.Params.SignedBlocksWindow, 64)
	if err != nil {
		return 0, 0, err
	}
	minSignedPerWindow, err = strconv.ParseFloat(result.Params.MinSignedPerWindow, 64)
	if err != nil {
		return 0, 0, err
	}
	return signedBlocksWindow, minSignedPerWindow, nil
}
