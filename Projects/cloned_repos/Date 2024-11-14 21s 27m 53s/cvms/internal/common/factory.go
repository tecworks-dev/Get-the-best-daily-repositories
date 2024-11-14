package common

import (
	"strconv"
	"time"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

const (
	Namespace                = "cvms"           // cosmos_validator_monitoring_service
	Subsystem                = "root"           // root subsytem for skip, health, ops
	Timeout                  = 13 * time.Second // root timeout for each package api call
	IndexerSQLDefaultTimeout = 10 * time.Second
)

var (
	// moniker is filter for ce
	Moniker string

	DefaultLabels = []string{ChainLabel, ChainIDLabel, PackageLabel, MainnetLabel, ErrLabel}

	// root skip counter for increasing count if any package is failed
	Skip = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "skip_counter"},
		DefaultLabels,
	)

	Health = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "health_checker"},
		DefaultLabels,
	)

	Ops = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: Namespace,
		Subsystem: Subsystem,
		Name:      "processed_ops_total"},
		DefaultLabels,
	)
)

func BuildRootLabels(p Packager) prometheus.Labels {
	return prometheus.Labels{
		ChainLabel:   p.ChainName,
		ChainIDLabel: p.ChainID,
		PackageLabel: p.Package,
		MainnetLabel: strconv.FormatBool(p.Mainnet),
		ErrLabel:     "",
	}
}

func BuildPackageLabels(p Packager) prometheus.Labels {
	tableChainID := helper.ParseToSchemaName(p.ChainID)
	return prometheus.Labels{
		ChainLabel:        p.ChainName,
		ChainIDLabel:      p.ChainID,
		TableChainIDLabel: tableChainID,
		PackageLabel:      p.Package,
		MainnetLabel:      strconv.FormatBool(p.Mainnet),
	}
}

func BuildPackageLabelsWithURL(p Packager, url string) prometheus.Labels {
	tableChainID := helper.ParseToSchemaName(p.ChainID)
	return prometheus.Labels{
		ChainLabel:        p.ChainName,
		ChainIDLabel:      p.ChainID,
		TableChainIDLabel: tableChainID,
		PackageLabel:      p.Package,
		BaseURLLabel:      url,
		MainnetLabel:      strconv.FormatBool(p.Mainnet),
	}
}
