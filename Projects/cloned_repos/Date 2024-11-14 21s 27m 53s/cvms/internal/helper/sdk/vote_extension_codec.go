package sdkhelper

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"fmt"
	"io"

	cometabci "github.com/cometbft/cometbft/abci/types"
	tmtypes "github.com/cometbft/cometbft/proto/tendermint/types"
	"github.com/cosmostation/cvms/internal/common/types"
	"github.com/klauspost/compress/zstd"
)

const VECommitFlag = tmtypes.BlockIDFlagCommit

// var BlockIDFlag_name = map[int32]string{
// 	0: "BLOCK_ID_FLAG_UNKNOWN",
// 	1: "BLOCK_ID_FLAG_ABSENT",
// 	2: "BLOCK_ID_FLAG_COMMIT",
// 	3: "BLOCK_ID_FLAG_NIL",
// }

func DecodingVoteExtensionTx(veTx string) ([]types.VoteExtension, error) {
	veBytes, err := base64.StdEncoding.DecodeString(veTx)
	if err != nil {
		return nil, err
	}
	bz, err := MustDecompress(veBytes)
	if err != nil {
		return nil, err
	}
	commitBz, err := DecodeExtendCommitInfo(bz)
	if err != nil {
		return nil, err
	}

	veList := make([]types.VoteExtension, 0)
	for _, vote := range commitBz.GetVotes() {
		// NOTE: https://github.com/cometbft/cometbft/blob/v0.38.x/proto/tendermint/types/validator.proto
		// ref; https://discord.com/channels/1010553709987639406/1235725015840985108/1296675516522168431
		var blockflag int64
		blockflag = int64(vote.GetBlockIdFlag())
		if len(vote.GetVoteExtension()) == 0 {
			blockflag = int64(tmtypes.BlockIDFlagUnknown)
		}
		veList = append(veList, types.VoteExtension{
			Address:         fmt.Sprintf("%X", vote.GetValidator().Address),
			BlockCommitFlag: blockflag,
			VoteExtension:   vote.GetVoteExtension(),
			Signature:       vote.GetExtensionSignature(),
		})
	}

	return veList, nil
}

func MustDecompress(bz []byte) ([]byte, error) {
	// Try zlib decompression first
	decompressedBz, err := zlibDecompress(bz)
	if err == nil {
		return decompressedBz, nil
	}

	// Fallback to zstd decompression
	return zstdDecompress(bz)
}

// ref; https://github.com/skip-mev/connect/blob/main/abci/strategies/codec/codec.go
func zlibDecompress(bz []byte) ([]byte, error) {
	if len(bz) == 0 {
		return nil, nil
	}
	r, err := zlib.NewReader(bytes.NewReader(bz))
	if err != nil {
		return nil, err
	}
	r.Close()

	// read bytes and return
	return io.ReadAll(r)
}

// ref; https://github.com/skip-mev/connect/blob/main/abci/strategies/codec/codec.go
func zstdDecompress(bz []byte) ([]byte, error) {
	dec, _ := zstd.NewReader(nil)
	return dec.DecodeAll(bz, nil)
}

// ref; https://github.com/skip-mev/connect/blob/main/abci/strategies/codec/codec.go
func DecodeExtendCommitInfo(bz []byte) (cometabci.ExtendedCommitInfo, error) {
	if len(bz) == 0 {
		return cometabci.ExtendedCommitInfo{}, nil
	}

	var extendedCommitInfo cometabci.ExtendedCommitInfo
	return extendedCommitInfo, extendedCommitInfo.Unmarshal(bz)
}

// for this feature, we should add dependency about connect; vetypes "github.com/skip-mev/slinky/abci/ve/types"
// NOTE: UNUSED FUNCTION
func DecodePrices(decodedBz []byte) (int, error) {
	// veCodec := compression.NewCompressionVoteExtensionCodec(
	// 	compression.NewDefaultVoteExtensionCodec(),
	// 	compression.NewZLibCompressor(),
	// )

	decompressedBz, err := zlibDecompress(decodedBz)
	if err != nil {
		return 0, err
	}

	_ = decompressedBz

	return 0, nil
}
