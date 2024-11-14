package exporter

import (
	"strconv"
	"strings"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/helper/config"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/sirupsen/logrus"
)

var PackageFilter string

func register(m common.Mode, f promauto.Factory, l *logrus.Logger, mc *config.MonitoringConfig, sc *config.SupportChains) error {
	l.Debugf("the package is filterd by package filter flag, only %s package is going to be register in exporter application", PackageFilter)
	for _, cc := range mc.ChainConfigs {
		chain := sc.Chains[cc.ChainID]
		mainnet := chain.Mainnet
		chainID := cc.ChainID
		chainName := chain.ChainName
		packages := chain.Packages
		protocolType := chain.ProtocolType
		isConsumer := chain.Consumer

		balanceDenom := chain.SupportAsset.Denom
		balanceDecimal := chain.SupportAsset.Decimal

		if len(cc.TrackingAddresses) > 0 {
			// NOTE: If there are tracking addresses in the config file,
			// 	enable balance package monitoring
			l.Debugf("found tracking address list: %v", cc.TrackingAddresses)
			packages = append(packages, "balance")
		}

		for _, pkg := range packages {
			// only register indexer packages among config packages
			if ok := helper.Contains(common.ExporterPackages, pkg); ok {
				if PackageFilter == "" {
					// all package is going to register
					err := selectPackage(m, f, l, mainnet, chainID, chainName, pkg, protocolType, balanceDenom, balanceDecimal, isConsumer, cc, mc.Monikers)
					if err != nil {
						l.WithField("package", pkg).WithField("chain", cc.ChainID).Errorf("this package is skipped by %s", err)
						common.Skip.With(prometheus.Labels{
							common.ChainLabel:   chainName,
							common.ChainIDLabel: chainID,
							common.PackageLabel: pkg,
							common.MainnetLabel: strconv.FormatBool(mainnet),
							common.ErrLabel:     err.Error(),
						}).Inc()
					}
				} else if strings.Contains(string(pkg), PackageFilter) {
					// specific package is going to register by 'PackageFilter' variable
					l.Debugf("filterd package %s for %s", pkg, chainName)
					err := selectPackage(m, f, l, mainnet, chainID, chainName, pkg, protocolType, balanceDenom, balanceDecimal, isConsumer, cc, mc.Monikers)
					if err != nil {
						l.WithField("package", pkg).WithField("chain", chainName).Infof("this package is skipped by %s", err)
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
	}
	return nil
}
