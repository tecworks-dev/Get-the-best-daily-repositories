package indexer

import (
	"fmt"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
)

var (
	// global variables for auto-register each packages in the custom registry
	registry = prometheus.NewRegistry()
	factory  = promauto.With(registry)
	router   = mux.NewRouter()
)

func buildPrometheusExporter(port string, logger *logrus.Logger) (*http.Server, promauto.Factory) {
	router.
		HandleFunc("/", defaultHandleFunction).
		Methods("GET")
	router.
		Handle("/metrics", buildPrometheusHandler(registry, logger)).
		Methods("GET")
	router.
		PathPrefix("/debug/pprof/").
		Handler(http.DefaultServeMux).
		Methods("GET")

	return &http.Server{
		Addr:    fmt.Sprintf(":%s", port),
		Handler: router,
	}, factory
}

func defaultHandleFunction(w http.ResponseWriter, r *http.Request) {
	w.Write([]byte(`<html>
	<head>
		<title> CVMS(Cosmos Validator Monitoring Service) </title>
		</head>
	<body>
		<h1>Cosmos Validator Monitoring Service - Indexer</h1>
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
