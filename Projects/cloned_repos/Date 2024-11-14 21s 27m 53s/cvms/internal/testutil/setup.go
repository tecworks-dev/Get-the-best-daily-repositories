package testutil

import (
	"database/sql"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper/logger"
	"github.com/go-resty/resty/v2"
	"github.com/joho/godotenv"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/sirupsen/logrus"
	"github.com/uptrace/bun"
	"github.com/uptrace/bun/dialect/pgdialect"
	"github.com/uptrace/bun/driver/pgdriver"
	"github.com/uptrace/bun/extra/bundebug"
)

var (
	// Configuration
	TestConfigPath string

	// Test Utils
	TestLogger *logrus.Logger

	// Test Prometheus factory
	TestFactory promauto.Factory
	TestHandler http.Handler

	// Test PostgresDB
	TestDB        *bun.DB
	TestIndexerDB common.IndexerDB
)

func SetupForTest() string {
	cmd := exec.Command("go", "env", "GOMOD")
	out, _ := cmd.Output()

	rootPath := strings.Split(string(out), "/go.mod")[0]
	envPath := fmt.Sprintf("%s/internal/testutil/%s", rootPath, ".env.test")

	// load .env.test variables
	godotenv.Load(envPath)

	// setup test paths
	TestConfigPath = fmt.Sprintf("%s/%s", rootPath, "config.yaml")
	TestLogger = logger.GetTestLogger()
	TestRegister := prometheus.NewRegistry()
	TestFactory = promauto.With(TestRegister)
	TestHandler = BuildTestHandler(TestRegister)

	// setup test db
	dsn := os.Getenv("TEST_DB_DNS")
	sqldb := sql.OpenDB(pgdriver.NewConnector(pgdriver.WithDSN(dsn)))
	sqldb.SetMaxOpenConns(1)

	TestDB = bun.NewDB(sqldb, pgdialect.New())
	TestDB.AddQueryHook(bundebug.NewQueryHook(
		bundebug.WithVerbose(true),
		bundebug.FromEnv("BUNDEBUG"),
	))

	indexerDB, _ := common.NewTestIndexerDB(dsn)
	TestIndexerDB = *indexerDB
	return ""
}

func BuildTestHandler(registry prometheus.Gatherer) http.Handler {
	return promhttp.HandlerFor(registry, promhttp.HandlerOpts{})
}

func GetTestExporter() *common.Exporter {
	l := logger.GetTestLogger()
	restyLogger := logrus.New()
	restyLogger.Out = io.Discard
	RPCClient := resty.New().
		SetRetryCount(10).
		SetRetryWaitTime(10 * time.Millisecond).
		SetRetryMaxWaitTime(3 * time.Second).
		SetLogger(restyLogger)
	APIClient := resty.New().
		SetRetryCount(10).
		SetRetryWaitTime(10 * time.Millisecond).
		SetRetryMaxWaitTime(3 * time.Second).
		SetLogger(restyLogger)
	GRPCClient := resty.New().
		SetRetryCount(10).
		SetRetryWaitTime(10 * time.Millisecond).
		SetRetryMaxWaitTime(3 * time.Second).
		SetLogger(restyLogger)
	entry := l.WithField("mode", "test")
	monikers := []string{"Cosmostation"}
	commonClient := common.CommonClient{
		RPCClient:  RPCClient,
		APIClient:  APIClient,
		GRPCClient: GRPCClient,
		Entry:      entry,
	}
	return &common.Exporter{
		CommonApp: common.CommonApp{
			CommonClient:   commonClient,
			EndPoint:       "",
			OptionalClient: common.CommonClient{},
		},
		Monikers: monikers,
	}
}

func BuildTestMetricName(namespace, subsystem, metricName string) string {
	return fmt.Sprintf(`%s_%s_%s`,
		namespace, subsystem, metricName,
	)
}

func CheckMetricsWithParams(
	checkMetric string,
	expectedMetricName, packageName, chainName string,
	additional ...string,
) (bool, string) {
	namePattern := regexp.MustCompile(expectedMetricName)
	packagePattern := regexp.MustCompile(fmt.Sprintf(`package="%s"`, packageName))
	chainPattern := regexp.MustCompile(fmt.Sprintf(`chain="%s"`, chainName))

	patterns := []string{namePattern.String(), packagePattern.String(), chainPattern.String()}

	if !namePattern.MatchString(checkMetric) || !packagePattern.MatchString(checkMetric) || !chainPattern.MatchString(checkMetric) {
		return false, strings.Join(patterns, ", ")
	}

	for _, addStr := range additional {
		if !regexp.MustCompile(addStr).MatchString(checkMetric) {
			patterns = append(patterns, addStr)
			return false, strings.Join(patterns, ", ")
		}
		patterns = append(patterns, addStr)
	}

	return true, strings.Join(patterns, ", ")
}
