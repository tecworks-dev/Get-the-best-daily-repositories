package exporter

import (
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper/config"
	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
)

func Build(port string, l *logrus.Logger, cfg *config.MonitoringConfig, sc *config.SupportChains) (
	/* prometheus exporter server */ *http.Server,
	/* unexpected error */ error,
) {
	var (
		// global variables for auto-register each packages in the custom registry
		registry = prometheus.NewRegistry()
		factory  = promauto.With(registry)
		router   = mux.NewRouter()
	)

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

	registry.MustRegister(common.Skip, common.Health, common.Ops)
	err := register(app, factory, l, cfg, sc)
	if err != nil {
		return nil, err
	}

	router.
		HandleFunc("/", defaultHandleFunction).
		Methods("GET")
	router.
		Handle("/metrics", buildPrometheusHandler(registry, l)).
		Methods("GET")
	router.
		PathPrefix("/debug/pprof/").
		Handler(http.DefaultServeMux).
		Methods("GET")

	return &http.Server{
		Addr:    fmt.Sprintf(":%s", port),
		Handler: router,
	}, nil
}

func defaultHandleFunction(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte(`<html>
	<head>
		<title> CVMS(Cosmos Validator Monitoring Service) </title>
		</head>
	<body>
		<h1>Cosmos Validator Monitoring Service - Exporter</h1>
		<h3><a href="https://cosmostation.io/">Prod by Cosmostation</a></h3>
		<p><a href="/metrics">Metrics</a></p>
	</body>
	</html>`))
}

func buildPrometheusHandler(registry prometheus.Gatherer, logger promhttp.Logger) http.Handler {
	return promhttp.HandlerFor(registry, promhttp.HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: promhttp.ContinueOnError,
		Timeout:       time.Second * 5,
	})
}
