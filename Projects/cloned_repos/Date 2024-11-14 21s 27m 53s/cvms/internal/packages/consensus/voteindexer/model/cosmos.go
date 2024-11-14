package model

import (
	"fmt"
	"time"

	"github.com/uptrace/bun"
)

// status := 0 is NaN(jailed or inactive) 1 is missed, 2 is voted, 3 is proposed
type ValidatorVote struct {
	bun.BaseModel         `bun:"table:voteindexer"`
	ID                    int64      `bun:"id,pk,autoincrement"`
	ChainInfoID           int64      `bun:"chain_info_id,pk,notnull"`
	Height                int64      `bun:"height,notnull"`
	ValidatorHexAddressID int64      `bun:"validator_hex_address_id,notnull"`
	Status                VoteStatus `bun:"status,notnull"`
	Timestamp             time.Time  `bun:"timestamp,notnull"`
}

func (vm ValidatorVote) String() string {
	return fmt.Sprintf("ValidatorVote<%d %d %d %d %d %d>",
		vm.ID,
		vm.ChainInfoID,
		vm.Height,
		vm.ValidatorHexAddressID,
		vm.Status,
		vm.Timestamp.Unix(),
	)
}

// Methods
type VoteStatus int

const (
	Missed VoteStatus = iota + 1
	Voted
	Proposed
)

type RecentValidatorVote struct {
	Moniker       string `bun:"moniker"`
	MaxHeight     int64  `bun:"max_height"`
	MinHeight     int64  `bun:"min_height"`
	ProposedCount int64  `bun:"proposed"`
	CommitedCount int64  `bun:"commited"`
	MissedCount   int64  `bun:"missed"`
}
