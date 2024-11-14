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

func Test_BalanceCollector(t *testing.T) {
	// setup
	_ = tests.SetupForTest()
	type testCase struct {
		testingName  string
		endpoints    common.Endpoints
		chainName    string
		protocolType string
		//  only balance
		balanceAddresses []string
		baseDenom        string
		baseExponent     int
	}

	testCases := []testCase{
		{
			testingName: "Kava Oracle Balance Check",
			endpoints: common.Endpoints{
				APIs: []string{os.Getenv("TEST_KAVA_ENDPOINT")},
			},
			chainName:        "kava",
			protocolType:     "cosmos",
			balanceAddresses: []string{os.Getenv("TEST_KAVA_ADDRESS")},
			baseDenom:        os.Getenv("TEST_KAVA_DENOM"),
			// baseExponent:     os.Getenv("TEST_KAVA_EXPONENT"),
			baseExponent: 6,
		},
		{
			testingName: "Injective Balances Check",
			endpoints: common.Endpoints{
				APIs: []string{os.Getenv("TEST_INJECTIVE_ENDPOINT_1"), os.Getenv("TEST_INJECTIVE_ENDPOINT_2")},
			},
			chainName:    "injective",
			protocolType: "cosmos",
			balanceAddresses: []string{
				os.Getenv("TEST_INJECTIVE_ADDRESS_1"),
				os.Getenv("TEST_INJECTIVE_ADDRESS_2"),
			},
			baseDenom: os.Getenv("TEST_INJECTIVE_DENOM"),
			// baseExponent: os.Getenv("TEST_INJECTIVE_EXPONENT"),
			baseExponent: 18,
		},
	}

	for _, tc := range testCases {
		if !assert.NotEqualValues(t, len(tc.endpoints.APIs), 0) {
			// validURLs is empty
			t.FailNow()
		}

		cc := config.ChainConfig{
			DisplayName: tc.chainName,
		}
		// build packager
		packager, err := common.NewPackager(common.NETWORK, tests.TestFactory, tests.TestLogger, true, "chainid", tc.chainName, Subsystem, tc.protocolType, cc, tc.endpoints)
		if err != nil {
			t.Fatal(err)
		}

		// set balance info
		packager.SetInfoForBalancePackage(tc.balanceAddresses, tc.baseDenom, tc.baseExponent)

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
		checkMetric := tests.BuildTestMetricName(common.Namespace, Subsystem, AmountMetricName)
		for _, address := range tc.balanceAddresses {
			passed, patterns := tests.CheckMetricsWithParams(string(body), checkMetric, Subsystem, tc.chainName, address)
			if !passed {
				t.Fatalf("Expected metric '%s' not found in response body: %s", checkMetric, string(body))
			}

			t.Logf("Check Metrics with these patterns: %s", patterns)
			t.Logf("Found expected metric: '%s' in response body", checkMetric)
			t.Logf("Actually Body:\n%s", body)
		}
	}
}
