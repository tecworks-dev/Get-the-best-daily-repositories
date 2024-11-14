package indexer

import (
	"errors"
	"net/http"
	"os"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper/config"
	dbhelper "github.com/cosmostation/cvms/internal/helper/db"
	"github.com/sirupsen/logrus"
)

func Build(port string, l *logrus.Logger, cfg *config.MonitoringConfig, sc *config.SupportChains) (
	/* prometheus indexer server */ *http.Server,
	/* unexpected error */ error,
) {
	// set application mode by monikers
	app := common.VALIDATOR
	for _, moniker := range cfg.Monikers {
		if moniker == "all" {
			app = common.NETWORK
		}
	}

	if app == common.VALIDATOR && len(cfg.Monikers) < 1 {
		return nil, errors.New("in cosmostation-exporter mode, you must add monike into the moniker flag")
	}

	// register root metircs
	registry.MustRegister(common.Skip, common.Health, common.Ops)

	// build prometheus server
	indexerServer, factory := buildPrometheusExporter(port, l)

	// create indexer DB
	dbCfg := common.IndexerDBConfig{
		Host:     os.Getenv("DB_HOST"),     // Get from environment variable DB_HOST
		Database: os.Getenv("DB_NAME"),     // Get from environment variable DB_NAME
		Port:     os.Getenv("DB_PORT"),     // Get from environment variable DB_PORT
		User:     os.Getenv("DB_USER"),     // Get from environment variable DB_USER
		Password: os.Getenv("DB_PASSWORD"), // Get from environment variable DB_PASSWORD
		Timeout:  30,
	}

	rt := os.Getenv("DB_RETENTION_PERIOD") // Get from environment variable DB_PASSWO
	if rt == "" {
		return nil, errors.New("DB_RETENTION_PERIOD is empty")
	}

	_, err := dbhelper.ParseRetentionPeriod(rt)
	if err != nil {
		return nil, err
	}

	idb, err := common.NewIndexerDB(dbCfg)
	if err != nil {
		return nil, err
	}
	idb.SetRetentionTime(rt)

	err = register(app, factory, l, idb, cfg, sc)
	if err != nil {
		return nil, err
	}

	return indexerServer, nil
}
