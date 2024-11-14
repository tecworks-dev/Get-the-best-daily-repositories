package db

import (
	"fmt"

	"github.com/cosmostation/cvms/internal/helper"
)

func MakePartitionTableName(indexName, chainID string) string {
	return fmt.Sprintf("public.%s_%s", indexName, helper.ParseToSchemaName(chainID))
}

func MakeCreatePartitionTableQuery(indexName, chainID string, chainInfoID int64) string {
	return fmt.Sprintf(
		`CREATE TABLE IF NOT EXISTS %s PARTITION OF "public"."%s" FOR VALUES IN ('%d');`,
		MakePartitionTableName(indexName, chainID), indexName, chainInfoID,
	)
}
