package indexer

import (
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/prometheus/client_golang/prometheus"
)

func (vidx *VEIndexer) initLabelsAndMetrics() {
	indexPointerBlockHeightMetric := vidx.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   subsystem,
		Name:        common.IndexPointerBlockHeightMetricName,
		ConstLabels: vidx.PackageLabels,
	})
	indexPointerBlockTimestampMetric := vidx.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   subsystem,
		Name:        common.IndexPointerBlockTimestampMetricName,
		ConstLabels: vidx.PackageLabels,
	})
	latestBlockHeightMetric := vidx.Factory.NewGauge(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   subsystem,
		Name:        common.LatestBlockHeightMetricName,
		ConstLabels: vidx.PackageLabels,
	})
	recentMissCounterMetric := vidx.Factory.NewGaugeVec(prometheus.GaugeOpts{
		Namespace:   common.Namespace,
		Subsystem:   subsystem,
		Name:        common.RecentMissCounterMetricName,
		ConstLabels: vidx.PackageLabels,
	}, []string{
		common.MonikerLabel,
	})

	indexPointerBlockHeightMetric.Set(0)
	vidx.MetricsMap[common.IndexPointerBlockHeightMetricName] = indexPointerBlockHeightMetric

	indexPointerBlockTimestampMetric.Set(0)
	vidx.MetricsMap[common.IndexPointerBlockTimestampMetricName] = indexPointerBlockTimestampMetric

	latestBlockHeightMetric.Set(0)
	vidx.MetricsMap[common.LatestBlockHeightMetricName] = latestBlockHeightMetric

	vidx.MetricsVecMap[common.RecentMissCounterMetricName] = recentMissCounterMetric
}

func (vidx *VEIndexer) updateRecentMissCounterMetric() {
	rveList, err := vidx.repo.SelectRecentValidatorExtensionVoteList(vidx.ChainID)
	if err != nil {
		vidx.Errorf("failed to update recent miss counter metric: %s", err)
	}

	for _, rve := range rveList {
		missCount := (rve.UnknownCount + rve.AbsentCount + rve.NilCount)
		vidx.MetricsVecMap[common.RecentMissCounterMetricName].
			With(prometheus.Labels{common.MonikerLabel: rve.Moniker}).
			Set(float64(missCount))
	}
}

func (vidx *VEIndexer) updatePrometheusMetrics(indexPointer int64, indexPointerTimestamp time.Time) {
	vidx.MetricsMap[common.IndexPointerBlockHeightMetricName].Set(float64(indexPointer))
	vidx.MetricsMap[common.IndexPointerBlockTimestampMetricName].Set((float64(indexPointerTimestamp.Unix())))
	vidx.Debugf("update prometheus metrics %d height", indexPointer)
}