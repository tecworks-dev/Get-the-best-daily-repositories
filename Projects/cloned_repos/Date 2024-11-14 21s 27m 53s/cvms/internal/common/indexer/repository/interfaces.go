package repository

import "github.com/cosmostation/cvms/internal/common/indexer/model"

// common repo interface
type IMetaRepository interface {
	IChainInfoRepository
	IIndexPointerRepository
	IValidatorInfoRepository

	// common sql interface for partition tables
	CreatePartitionTable(IndexName, chainID string) error
	InitPartitionTablesByChainInfoID(IndexName, chainID string, latestHeight int64) error
}

// interface for about meta.chain_info table
type IChainInfoRepository interface {
	InsertChainInfo(chainName, chainID string, IsMainnet bool) (int64, error)
	SelectChainInfoIDByChainID(chainID string) (int64, error)
}

// interface for about meta.index_pointer table
type IIndexPointerRepository interface {
	InitializeIndexPointerByChainID(indexTableName, chainID string, startHeight int64) error
	GetLastIndexPointerByIndexTableName(indexTableName string, chainInfoID int64) (model.IndexPointer, error)
	CheckIndexpoinerAlreadyInitialized(indexTableName string, chainInfoID int64) (bool, error)
}

// interface for about meta.validator_info table
type IValidatorInfoRepository interface {
	CreateValidatorInfoPartitionTableByChainID(chainID string) error
	GetValidatorInfoListByChainInfoID(chainInfoID int64) (validatorInfoList []model.ValidatorInfo, err error)
	InsertValidatorInfoList(validatorInfoList []model.ValidatorInfo) error
	GetValidatorInfoListByMonikers(chainInfoID int64, monikers []string) ([]model.ValidatorInfo, error)
}
