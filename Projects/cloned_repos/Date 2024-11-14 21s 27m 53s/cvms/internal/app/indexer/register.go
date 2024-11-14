package indexer

import (
	"strconv"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/helper/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"
)

func register(m common.Mode, f promauto.Factory, l *logrus.Logger, idb *common.IndexerDB, mc *config.MonitoringConfig, sc *config.SupportChains) error {
	l.Infof("supported packages for indexer application: %v", common.IndexPackages)
	for _, cc := range mc.ChainConfigs {
		chain := sc.Chains[cc.ChainID]
		mainnet := chain.Mainnet
		chainID := cc.ChainID
		chainName := chain.ChainName
		packages := chain.Packages
		protocolType := chain.ProtocolType
		isConsumer := chain.Consumer

		for _, pkg := range packages {
			// only register indexer packages among config packages
			if ok := helper.Contains(common.IndexPackages, pkg); ok {
				// all package is going to register
				err := selectPackage(m, f, l, idb, mainnet, chainID, chainName, pkg, protocolType, isConsumer, cc, mc.Monikers)
				if err != nil {
					l.WithField("package", pkg).WithField("chain", chainName).WithField("chain_id", chainID).
						Errorf("this package was failed to start while initiating, so that the package will be skipped: %s", err)

					common.Skip.With(prometheus.Labels{
						common.ChainLabel:   chainName,
						common.ChainIDLabel: chainID,
						common.PackageLabel: pkg,
						common.MainnetLabel: strconv.FormatBool(mainnet),
						common.ErrLabel:     err.Error(),
					}).Inc()
				}
			}
		}
	}
	return nil
}
