package collector

import (
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper/config"
	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/stretchr/testify/assert"
)

type testCase struct {
	testingName  string
	endpoints    common.Endpoints
	chainName    string
	protocolType string
}

var (
	testMoniker string
	testCases   []testCase
	testMetrics = []string{
		NonceMetricName,
		HeighestNonceMetricName,
	}
)

func TestMain(m *testing.M) {
	_ = tests.SetupForTest()
	testMoniker = os.Getenv("TEST_MONIKER")
	testCases = []testCase{
		{
			testingName:  "Injective Eventnonce Package",
			endpoints:    common.Endpoints{GRPCs: []string{os.Getenv("TEST_INJECTIVE_GRPC_ENDPOINT")}},
			chainName:    "injective",
			protocolType: "cosmos",
		},
		{
			testingName:  "Gravity Bridge Eventnonce Package",
			endpoints:    common.Endpoints{GRPCs: []string{os.Getenv("TEST_GRAVITY_GRPC_ENDPOINT")}},
			chainName:    "gravity-bridge",
			protocolType: "cosmos",
		},
		{
			testingName:  "Sommelier Eventnonce Package",
			endpoints:    common.Endpoints{GRPCs: []string{os.Getenv("TEST_SOMMELIER_GRPC_ENDPOINT")}},
			chainName:    "sommelier",
			protocolType: "cosmos",
		},
	}
	os.Exit(m.Run())
}

func TestEventnoncePackageInVMSMode(t *testing.T) {
	for _, tc := range testCases {
		if !assert.NotEqualValues(t, len(tc.endpoints.GRPCs), 0) {
			// validURLs is empty
			t.FailNow()
		}

		cc := config.ChainConfig{
			DisplayName: tc.chainName,
		}

		// build packager
		packager, err := common.NewPackager(common.NETWORK, tests.TestFactory, tests.TestLogger, true, "chainid", cc.DisplayName, Subsystem, tc.protocolType, cc, tc.endpoints)
		if err != nil {
			t.Fatal(err)
		}

		err = Start(*packager)
		assert.NoErrorf(t, err, "error message %s", "formatted")
	}

	// sleep 3sec for waiting collecting data from nodes
	time.Sleep(3 * time.Second)

	// Create a new HTTP test server
	server := httptest.NewServer(tests.TestHandler)
	defer server.Close()

	// Make a request to the test server
	resp, err := http.Get(server.URL)
	if err != nil {
		t.Fatalf("Failed to make GET request: %v", err)
	}
	defer resp.Body.Close()

	// Check the HTTP response status code
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status code 200, got %d", resp.StatusCode)
	}

	// Check the response body for the expected metric
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	for _, tc := range testCases {
		t.Logf("Test Name: %s", tc.testingName)
		for _, metricName := range testMetrics {
			checkMetric := tests.BuildTestMetricName(common.Namespace, Subsystem, metricName)
			passed, patterns := tests.CheckMetricsWithParams(string(body), checkMetric, Subsystem, tc.chainName)
			if !passed {
				t.Fatalf("Expected metric '%s' not found in response body: %s", checkMetric, string(body))
			}

			t.Logf("Check Metrics with these patterns: %s", patterns)
			t.Logf("Found expected metric: '%s' in response body", checkMetric)
			// t.Logf("Actually Body:\n%s", body)
		}
	}
}

func TestEventnoncePackageInSoloMode(t *testing.T) {
	// additional setup
	mode := common.VALIDATOR
	r := prometheus.NewRegistry()
	f := promauto.With(r)
	h := tests.BuildTestHandler(r)

	for _, tc := range testCases {
		if !assert.NotEqualValues(t, len(tc.endpoints.GRPCs), 0) {
			// validURLs is empty
			t.FailNow()
		}

		cc := config.ChainConfig{
			DisplayName: tc.chainName,
		}
		// build packager
		packager, err := common.NewPackager(mode, f, tests.TestLogger, true, "chainid", cc.DisplayName, Subsystem, tc.protocolType, cc, tc.endpoints, testMoniker)
		if err != nil {
			t.Fatal(err)
		}

		err = Start(*packager)
		assert.NoErrorf(t, err, "error message %s", "formatted")
	}

	// sleep 3sec for waiting collecting data from nodes
	time.Sleep(3 * time.Second)

	// Create a new HTTP test server
	server := httptest.NewServer(h)
	defer server.Close()

	// Make a request to the test server
	resp, err := http.Get(server.URL)
	if err != nil {
		t.Fatalf("Failed to make GET request: %v", err)
	}
	defer resp.Body.Close()

	// Check the HTTP response status code
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("Expected status code 200, got %d", resp.StatusCode)
	}

	// Check the response body for the expected metric
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}

	for _, tc := range testCases {
		t.Logf("Test Name: %s", tc.testingName)
		for _, metricName := range testMetrics {
			checkMetric := tests.BuildTestMetricName(common.Namespace, Subsystem, metricName)
			passed, patterns := tests.CheckMetricsWithParams(string(body), checkMetric, Subsystem, tc.chainName)
			if !passed {
				t.Fatalf("Expected metric '%s' not found in response body: %s", checkMetric, string(body))
			}

			t.Logf("Check Metrics with these patterns: %s", patterns)
			t.Logf("Found expected metric: '%s' in response body", checkMetric)
			// t.Logf("Actually Body:\n%s", body)
		}
	}
}
