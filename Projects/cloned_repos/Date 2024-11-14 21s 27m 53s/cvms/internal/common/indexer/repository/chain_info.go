package repository

import (
	"context"

	"github.com/cosmostation/cvms/internal/common/indexer/model"
	"github.com/pkg/errors"
)

func (repo *MetaRepository) InsertChainInfo(chainName, chainID string, isMainnet bool) (int64, error) {
	ctx := context.Background()
	defer ctx.Done()

	chainInfo := &model.ChainInfo{
		ChainName: chainName,
		Mainnet:   isMainnet,
		ChainID:   chainID,
	}

	_, err := repo.
		NewInsert().
		Model(chainInfo).
		ExcludeColumn("id").
		Returning("*").
		Exec(ctx)

	if err != nil {
		return 0, errors.Wrapf(err, "failed to insert new chain_info")
	}

	return chainInfo.ID, nil
}

func (repo *MetaRepository) SelectChainInfoIDByChainID(chainID string) (int64, error) {
	ctx := context.Background()
	defer ctx.Done()

	chainInfo := &model.ChainInfo{}
	err := repo.
		NewSelect().
		ModelTableExpr("meta.chain_info").
		ColumnExpr("*").
		Where("chain_id = ?", chainID).
		Scan(ctx, chainInfo)
	if err != nil {
		return 0, err
	}

	return chainInfo.ID, nil
}
