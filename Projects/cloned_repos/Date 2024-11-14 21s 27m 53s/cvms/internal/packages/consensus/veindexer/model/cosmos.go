package model

import (
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

type ValidatorExtensionVote struct {
	bun.BaseModel         `bun:"table:veindexer"`
	ID                    int64     `bun:"id,pk,autoincrement"`
	ChainInfoID           int64     `bun:"chain_info_id,pk,notnull"`
	Height                int64     `bun:"height,notnull"`
	ValidatorHexAddressID int64     `bun:"validator_hex_address_id,notnull"`
	Status                int64     `bun:"status,notnull"`
	VELength              int       `bun:"vote_extension_length,notnull"`
	Timestamp             time.Time `bun:"timestamp,notnull"`
}

func (vev ValidatorExtensionVote) String() string {
	return fmt.Sprintf("ValidatorExtensionVote<%d %d %d %d %d %d %d>",
		vev.ID,
		vev.ChainInfoID,
		vev.Height,
		vev.ValidatorHexAddressID,
		vev.Status,
		vev.VELength,
		vev.Timestamp.Unix(),
	)
}

type RecentValidatorExtensionVote struct {
	Moniker      string `bun:"moniker"`
	MaxHeight    int64  `bun:"max_height"`
	MinHeight    int64  `bun:"min_height"`
	UnknownCount int64  `bun:"unknown"`
	AbsentCount  int64  `bun:"absent"`
	CommitCount  int64  `bun:"commit"`
	NilCount     int64  `bun:"nil"`
}
