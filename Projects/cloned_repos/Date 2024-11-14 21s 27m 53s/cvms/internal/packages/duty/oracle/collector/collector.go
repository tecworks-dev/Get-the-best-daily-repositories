package collector

import (
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper/healthcheck"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/router"
	"github.com/pkg/errors"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/duty/oracle/types"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	_ common.CollectorStart = Start
	_ common.CollectorLoop  = loop
)

// NOTE: this is for solo mode
var packageMonikers []string

const (
	Subsystem      = "oracle"
	SubsystemSleep = 15 * time.Second
	UnHealthSleep  = 10 * time.Second

	MissCounterMetricName       = "miss_counter"
	SlashWindowMetricName       = "slash_window"
	VotePeriodMetricName        = "vote_period"
	MinValidPerWindowMetricName = "min_valid_per_window"
	VoteWindowMetricName        = "vote_window"
	BlockHeightMetricName       = "block_height"
)

func Start(p common.Packager) error {
	if ok := helper.Contains(types.SupportedChains, p.ChainName); ok {
		packageMonikers = p.Monikers
		for _, api := range p.APIs {
			client := common.NewExporter(p)
			client.SetAPIEndPoint(api)
			go loop(client, p)
			break
		}
		return nil
	}
	return errors.Errorf("unsupprted chain: %s", p.ChainName)
}

func loop(c *common.Exporter, p common.Packager) {
	rootLabels := common.BuildRootLabels(p)
	packageLabels := common.BuildPackageLabels(p)

	// each validators
	missCounterMetric := p.Factory.NewGaugeVec(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        MissCounterMetricName,
		ConstLabels: packageLabels,
	}, []string{common.ValidatorAddressLabel, common.MonikerLabel},
	)

	// each chain
	slashWindowMetric := p.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        SlashWindowMetricName,
		ConstLabels: packageLabels,
	})
	votePeriodMetric := p.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        VotePeriodMetricName,
		ConstLabels: packageLabels,
	})
	minValidPerWindowMetric := p.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        MinValidPerWindowMetricName,
		ConstLabels: packageLabels,
	})
	voteWindowMetric := p.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        VoteWindowMetricName,
		ConstLabels: packageLabels,
	})
	blockHeightMetric := p.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   Subsystem,
		Name:        BlockHeightMetricName,
		ConstLabels: packageLabels,
	})

	isUnhealth := false
	for {
		// node health check
		if isUnhealth {
			healthEndpoints := healthcheck.FilterHealthEndpoints(p.APIs, p.ProtocolType)
			for _, endpoint := range healthEndpoints {
				c.SetAPIEndPoint(endpoint)
				c.Infoln("client endpoint will be changed with health endpoint for this package")
				isUnhealth = false
				break
			}
			if len(healthEndpoints) == 0 {
				c.Errorln("failed to get any health endpoints from healthcheck filter, retry sleep 10s")
				time.Sleep(UnHealthSleep)
				continue
			}
		}

		// collect status
		status, err := router.GetStatus(c, p.ChainName)
		if err != nil {
			common.Health.With(rootLabels).Set(0)
			common.Ops.With(rootLabels).Inc()
			isUnhealth = true

			c.Logger.Errorf("failed to update metrics: %s", err.Error())
			time.Sleep(SubsystemSleep)

			continue
		}

		c.Debugf("got total %d validators status for oracle package", len(status.Validators))

		if p.Mode == common.NETWORK {
			// update metrics for each validators
			for _, item := range status.Validators {
				missCounterMetric.
					With(prometheus.Labels{
						common.ValidatorAddressLabel: item.ValidatorOperatorAddress,
						common.MonikerLabel:          item.Moniker,
					}).
					Set(float64(item.MissCounter))
			}
		} else {
			// filter metrics for only specific validator
			for _, item := range status.Validators {
				if ok := helper.Contains(packageMonikers, item.Moniker); ok {
					missCounterMetric.
						With(prometheus.Labels{
							common.ValidatorAddressLabel: item.ValidatorOperatorAddress,
							common.MonikerLabel:          item.Moniker,
						}).
						Set(float64(item.MissCounter))
				}
			}
		}

		// update metrics by each chain
		slashWindowMetric.Set(status.SlashWindow)
		votePeriodMetric.Set(status.VotePeriod)
		minValidPerWindowMetric.Set(status.MinimumValidPerWindow)
		voteWindowMetric.Set(status.VoteWindow)
		blockHeightMetric.Set(status.BlockHeight)

		c.Infof("updated metrics successfully and going to sleep %s ...", SubsystemSleep.String())

		// update health and ops
		common.Health.With(rootLabels).Set(1)
		common.Ops.With(rootLabels).Inc()

		// sleep
		time.Sleep(SubsystemSleep)
	}
}
