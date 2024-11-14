package router

import (
	"os"
	"testing"

	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestRequestMissCounter(t *testing.T) {
	_ = tests.SetupForTest()
	testMoniker := os.Getenv("TEST_MONIKER")
	TestCases := []struct {
		testingName string
		chainName   string
		endpoint    string
	}{
		{
			testingName: "Umee Oracle Status Check",
			chainName:   "umee",
			endpoint:    os.Getenv("TEST_UMEE_ENDPOINT"),
		},
		{
			testingName: "Sei Oracle Status Check",
			chainName:   "sei",
			endpoint:    os.Getenv("TEST_SEI_ENDPOINT"),
		},
		{
			testingName: "Nibiru Oracle Status Check",
			chainName:   "nibiru",
			endpoint:    os.Getenv("TEST_NIBIRU_ENDPOINT"),
		},
	}

	for _, tc := range TestCases {
		exporter := tests.GetTestExporter()
		t.Run(tc.testingName, func(t *testing.T) {
			if !assert.NotEqualValues(t, tc.endpoint, "") {
				// endpoint is empty
				t.FailNow()
			}
			// setup
			exporter.SetAPIEndPoint(tc.endpoint)
			// get status
			status, err := GetStatus(exporter, tc.chainName)
			if err != nil {
				t.Fatalf("%s: %s", tc.testingName, err.Error())
			}

			// assert.NotNil(t, status)
			for _, item := range status.Validators {
				if item.Moniker == testMoniker {
					t.Log("got our test validator information for oracle package")
					t.Logf("valoper address: %s", item.ValidatorOperatorAddress)
					t.Logf("miss_counter: %d", item.MissCounter)
				}
			}
			t.Logf("block height: %.2f", status.BlockHeight)
			t.Logf("min_valid_per_window: %.2f", status.MinimumValidPerWindow)
			t.Logf("slash_window: %.f", status.SlashWindow)
			t.Logf("vote_period: %.f", status.VotePeriod)
			t.Logf("vote_window: %.f", status.VoteWindow)
		})
	}
}
