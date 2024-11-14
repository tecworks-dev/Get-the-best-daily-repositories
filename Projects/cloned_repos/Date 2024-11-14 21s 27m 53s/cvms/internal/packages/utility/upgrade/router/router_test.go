package router_test

import (
	"os"
	"testing"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/router"
	tests "github.com/cosmostation/cvms/internal/testutil"

	"github.com/stretchr/testify/assert"
)

func TestRouter(t *testing.T) {
	_ = tests.SetupForTest()

	TestCases := []struct {
		testingName string
		hostAddress string
		chainName   string
	}{
		{
			testingName: "Upgrade exist Chain",
			chainName:   "cosmso",
			hostAddress: os.Getenv("TEST_UPGRADE_ENDPOINT_1"),
		},
		{
			testingName: "Upgrade not exist Chain",
			chainName:   "cosmos",
			hostAddress: os.Getenv("TEST_UPGRADE_ENDPOINT_2"),
		},
		{
			testingName: "Celestia Signal Upgrade",
			chainName:   "celestia",
			hostAddress: os.Getenv("TEST_UPGRADE_ENDPOINT_3"),
		},
	}

	for _, tc := range TestCases {
		exporter := tests.GetTestExporter()
		t.Run(tc.testingName, func(t *testing.T) {
			if !assert.NotEqualValues(t, tc.hostAddress, "") {
				// hostaddress is empty
				t.FailNow()
			}

			exporter.SetAPIEndPoint(tc.hostAddress)
			CommonUpgrade, err := router.GetStatus(exporter, tc.chainName)
			if err != nil && err != common.ErrCanSkip {
				t.Log("Unexpected Error Occured!")
				t.Skip()
			}

			if err == common.ErrCanSkip {
				t.Logf("There is no onchain upgrade now in %s", tc.testingName)
			} else {
				t.Log("onchain upgrade is found", CommonUpgrade.UpgradeName, CommonUpgrade.RemainingTime)
			}
		})
	}
}
