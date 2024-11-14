package exporter

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

func Test_PrometheusRegsiter(t *testing.T) {
	// Global variables for auto-register each package in the custom registry
	tRegistry := prometheus.NewRegistry()
	tFactory := promauto.With(tRegistry)

	// Define a CounterVec metric
	httpRequestsTotal := tFactory.NewCounterVec(
		prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "Number of HTTP requests.",
		},
		[]string{"method", "path"},
	)

	// Register the metric with the custom registry
	// tRegistry.MustRegister(httpRequestsTotal)

	// increase counter
	httpRequestsTotal.WithLabelValues("GET", "/test").Inc()
	httpRequestsTotal.WithLabelValues("GET", "/test").Inc()

	// httpRequestsTotal.Reset()

	// Gather the metrics from the custom registry
	metricFamilies, err := tRegistry.Gather()
	if err != nil {
		panic(err)
	}

	// Check if our metric is present in the gathered metrics
	found := false
	for _, m := range metricFamilies {

		for _, v := range m.GetMetric() {
			t.Log(v.GetCounter())
		}
		t.Log(m.GetName())
		if *m.Name == "http_requests_total" {
			found = true
			break
		}
	}

	if found {
		t.Log("Metric 'http_requests_total' found in the registry!")
	} else {
		t.Log("Metric 'http_requests_total' not found in the registry.")
	}

	// unregister
	tRegistry.Unregister(httpRequestsTotal)
	// t.Log(ok)

	// Gather the metrics from the custom registry
	metricFamilies, err = tRegistry.Gather()
	if err != nil {
		panic(err)
	}

	// Check if our metric is present in the gathered metrics
	found = false
	for _, m := range metricFamilies {

		for _, v := range m.GetMetric() {
			t.Log(v.GetCounter())
		}
		t.Log(m.GetName())
		if *m.Name == "http_requests_total" {
			found = true
			break
		}
	}

	if found {
		t.Log("Metric 'http_requests_total' found in the registry!")
	} else {
		t.Log("Metric 'http_requests_total' not found in the registry.")
	}
}
