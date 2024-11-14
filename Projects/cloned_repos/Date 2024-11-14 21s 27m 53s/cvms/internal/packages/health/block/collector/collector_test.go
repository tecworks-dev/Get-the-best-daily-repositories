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
	"github.com/stretchr/testify/assert"
)

func TestBlockPackage(t *testing.T) {
	// setup
	_ = tests.SetupForTest()

	type testCase struct {
		testingName  string
		chainName    string
		protocolType string
		endpoints    common.Endpoints
	}

	testCases := []testCase{
		{
			testingName:  "Cosmos Block Collector Test",
			chainName:    "cosmos",
			protocolType: "cosmos",
			endpoints:    common.Endpoints{RPCs: []string{os.Getenv("TEST_COSMOS_HOST_ADDRESS")}},
		},
		{
			testingName:  "Ethereum Block Collector Test",
			chainName:    "ethereum",
			protocolType: "ethereum",
			endpoints:    common.Endpoints{RPCs: []string{os.Getenv("TEST_ETHEREUM_HOST_ADDRESS")}},
		},
		{
			testingName:  "Celestia Block Collector Test",
			chainName:    "celestia",
			protocolType: "celestia",
			endpoints:    common.Endpoints{RPCs: []string{os.Getenv("TEST_CELESTIA_HOST_ADDRESS")}},
		},
		{
			testingName:  "Bera Block Collector Testing",
			chainName:    "bera",
			protocolType: "ethereum",
			endpoints:    common.Endpoints{RPCs: []string{os.Getenv("TEST_BERA_HOST_ADDRESS")}},
		},
	}

	for _, tc := range testCases {
		cc := config.ChainConfig{
			DisplayName: tc.chainName,
		}

		// build packager
		packager, err := common.NewPackager(common.NETWORK, tests.TestFactory, tests.TestLogger, true, "chainid", tc.chainName, Subsystem, tc.protocolType, cc, tc.endpoints)
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

	testMetrics := []string{
		BlockHeightMetricName,
		TimestampMetricName,
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
