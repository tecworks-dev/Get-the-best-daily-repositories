package common

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/pkg/errors"

	"github.com/jinzhu/inflection"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
	"github.com/uptrace/bun/extra/bundebug"
	"github.com/uptrace/bun/schema"
)

type IndexerDB struct {
	*bun.DB
	RetentionPeriod string
}

type IndexerDBConfig struct {
	Host     string `toml:"host"`
	Database string `toml:"database"`
	Port     string `toml:"port"`
	User     string `toml:"user"`
	Password string `toml:"password"`
	Timeout  int64  `toml:"db_timeout"`
}

func NewIndexerDB(cfg IndexerDBConfig) (*IndexerDB, error) {
	if cfg == (IndexerDBConfig{}) || cfg.User == "" {
		return nil, errors.New("you should provide DB envs like DB_HOST, DB_PORT...")
	}

	timeout := cfg.Timeout
	if cfg.Timeout == 0 {
		timeout = 10
	}

	// setup variables for db connection
	dsn := fmt.Sprintf(
		"postgres://%s:%s@%s:%s/%s?sslmode=disable",
		cfg.User, cfg.Password, cfg.Host, cfg.Port, cfg.Database,
	)
	timeoutDuration := time.Second * time.Duration(timeout)

	// initiate db
	sqldb := sql.OpenDB(pgdriver.NewConnector(
		pgdriver.WithDSN(dsn),
		pgdriver.WithTimeout(timeoutDuration),
		pgdriver.WithDialTimeout(timeoutDuration),
		pgdriver.WithReadTimeout(timeoutDuration),
	))

	db := bun.NewDB(sqldb, pgdialect.New())
	db.AddQueryHook(bundebug.NewQueryHook(
		// disable the hook
		bundebug.WithEnabled(false),
		bundebug.WithVerbose(false),
		// BUNDEBUG=1 logs failed queries
		// BUNDEBUG=2 logs all queries
		bundebug.FromEnv("BUNDEBUG"),
	))

	err := db.Ping()
	if err != nil {
		return nil, fmt.Errorf("failed to ping: %s", err)
	}

	schema.SetTableNameInflector(inflection.Singular)

	return &IndexerDB{
		DB: db,
	}, nil
}

func (db *IndexerDB) SetRetentionTime(retentionPeriod string) {
	db.RetentionPeriod = retentionPeriod
}

// TODO: currently, we don't use this helper.DB
func (db *IndexerDB) CloseConn() error {
	return db.DB.Close()
}

func NewTestIndexerDB(dsn string) (*IndexerDB, error) {
	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(dsn)))
	sqldb.SetMaxOpenConns(1)

	db := bun.NewDB(sqldb, pgdialect.New())
	db.AddQueryHook(bundebug.NewQueryHook(
		bundebug.WithVerbose(true),
		bundebug.FromEnv("BUNDEBUG"),
	))

	// NOTE: No need to ping in test mode
	// err := db.Ping()
	// if err != nil {
	// 	return nil, fmt.Errorf("failed to ping: %s", err)
	// }

	schema.SetTableNameInflector(inflection.Singular)

	return &IndexerDB{
		DB: db,
	}, nil
}
