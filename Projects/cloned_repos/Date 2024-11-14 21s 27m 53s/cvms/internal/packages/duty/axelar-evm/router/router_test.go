package router

import (
	"os"
	"testing"

	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestGetStatus(t *testing.T) {
	_ = tests.SetupForTest()
	testMoniker := os.Getenv("TEST_MONIKER")
	TestCases := []struct {
		testingName    string
		chainName      string
		endpoint       string
		expectedResult float64
	}{
		{
			testingName:    "Axelar EVM Chains Register Status Testing",
			chainName:      "axelar",
			endpoint:       os.Getenv("TEST_AXELAR_ENDPOINT"),
			expectedResult: 1,
		},
	}

	for _, tc := range TestCases {
		client := tests.GetTestExporter()
		t.Run(tc.testingName, func(t *testing.T) {
			if !assert.NotEqualValues(t, tc.endpoint, "") {
				// endpoint is empty
				t.FailNow()
			}

			// setup
			client.SetAPIEndPoint(tc.endpoint)
			// get status
			status, err := GetStatus(client, tc.chainName)
			if err != nil {
				t.Fatalf("%s: %s", tc.testingName, err.Error())
			}

			for _, status := range status.Validators {
				if status.Moniker == testMoniker {
					actual := status.Status
					if tc.expectedResult != actual {
						t.Logf("Expected %.f does not match actual %.f", tc.expectedResult, actual)
					}
					t.Logf("Matched status %s is registered now", status.EVMChainName)
				}
			}
			assert.NotEmpty(t, status)
		})
	}
}
