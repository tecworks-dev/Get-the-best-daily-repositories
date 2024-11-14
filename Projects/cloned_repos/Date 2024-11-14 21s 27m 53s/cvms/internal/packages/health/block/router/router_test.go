package router_test

import (
	"os"
	"testing"
	"time"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/health/block/router"
	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestRequestBlockTimeStamp(t *testing.T) {
	_ = tests.SetupForTest()

	type testCase struct {
		testingName string
		chainType   string
		hostAddress string
	}

	testCases := []testCase{
		{
			testingName: "CosomsChainType",
			chainType:   "cosmos",
			hostAddress: os.Getenv("TEST_COSMOS_HOST_ADDRESS"),
		},
		{
			testingName: "EthereumChainType",
			chainType:   "ethereum",
			hostAddress: os.Getenv("TEST_ETHEREUM_HOST_ADDRESS"),
		},
		{
			testingName: "Celestia Block Package Testing",
			chainType:   "celestia",
			hostAddress: os.Getenv("TEST_CELESTIA_HOST_ADDRESS"),
		},
		{
			testingName: "Bera Block Package Testing",
			chainType:   "ethereum",
			hostAddress: os.Getenv("TEST_BERA_HOST_ADDRESS"),
		},
	}

	for _, tc := range testCases {
		timestamp := uint64(time.Now().Unix())
		exporter := tests.GetTestExporter()

		t.Run(tc.testingName, func(t *testing.T) {
			if tc.hostAddress == "" {
				t.Logf("%s has empty base URL. check .env.test file", tc.testingName)
				t.SkipNow()
			}

			if !helper.ValidateURL(tc.hostAddress) {
				t.Logf("%s is unvalidate endpoint for block package", tc.testingName)
				t.SkipNow()
			}

			exporter.SetRPCEndPoint(tc.hostAddress)
			exporter.SetAPIEndPoint(tc.hostAddress)
			status, err := router.GetStatus(exporter, tc.chainType)
			if err != nil {
				t.Logf("%s: %s", tc.testingName, err.Error())
				t.FailNow()
			}

			t.Logf("latest block height: %d", int(status.LastBlockHeight))
			t.Logf("latest block timestamp: %d", int(status.LastBlockTimeStamp))

			timeInterval := int(float64(timestamp) - status.LastBlockTimeStamp)
			assert.LessOrEqualf(t, timeInterval, 300, "error message: %v", timeInterval)
		})
	}
}
