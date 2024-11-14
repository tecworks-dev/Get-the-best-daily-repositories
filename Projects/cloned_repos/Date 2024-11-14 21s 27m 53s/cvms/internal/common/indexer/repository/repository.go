package repository

import (
	"context"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/common/indexer/model"
	dbhelper "github.com/cosmostation/cvms/internal/helper/db"
	"github.com/pkg/errors"
	"github.com/uptrace/bun"
)

type MetaRepository struct {
	defaultTimeout time.Duration
	*bun.DB
}

// New general repository constructor
func NewMetaRepository(indexerDB common.IndexerDB) IMetaRepository {
	return &MetaRepository{common.IndexerSQLDefaultTimeout, indexerDB.DB}
}

// Because of using chainInfoID, some users can use wrong chain_info_id about input chainID
func (repo *MetaRepository) CreatePartitionTable(IndexName, chainID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), repo.defaultTimeout)
	defer cancel()

	chainInfo := &model.ChainInfo{}
	err := repo.
		NewSelect().
		Model(chainInfo).
		Column("id").
		Where("chain_id = ?", chainID).
		Scan(ctx)
	if err != nil {
		return errors.Wrapf(err, "failed to select chain_info id by chain_id")
	}

	_, err = repo.NewRaw(dbhelper.MakeCreatePartitionTableQuery(IndexName, chainID, chainInfo.ID)).Exec(ctx)
	if err != nil {
		return errors.Wrap(err, "failed to create a new partition table")
	}

	return nil
}

func (repo *MetaRepository) InitPartitionTablesByChainInfoID(indexName, chainID string, latestHeight int64) error {
	err := repo.CreateValidatorInfoPartitionTableByChainID(chainID)
	if err != nil {
		return errors.Wrapf(err, "failed to partition table for meta.validator_info")
	}

	err = repo.CreatePartitionTable(indexName, chainID)
	if err != nil {
		return errors.Wrapf(err, "failed to partition table for public.voteindexer")
	}

	err = repo.InitializeIndexPointerByChainID(indexName, chainID, latestHeight)
	if err != nil {
		return errors.Wrapf(err, "failed to init index pointer table for meta.index_pointer")
	}

	return nil
}
