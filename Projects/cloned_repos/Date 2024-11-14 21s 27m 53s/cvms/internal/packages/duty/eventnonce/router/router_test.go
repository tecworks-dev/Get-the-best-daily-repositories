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
			testingName:    "InjectiveEventnonceStatusTesting",
			chainName:      "injective",
			endpoint:       os.Getenv("TEST_INJECTIVE_GRPC_ENDPOINT"),
			expectedResult: 0,
		},
		{
			testingName:    "GravityBridgeEventnoncePackageTesting",
			chainName:      "gravity-bridge",
			endpoint:       os.Getenv("TEST_GRAVITY_GRPC_ENDPOINT"),
			expectedResult: 0,
		},
		{
			testingName:    "SommelierEventnoncePackageTesting",
			chainName:      "sommelier",
			endpoint:       os.Getenv("TEST_SOMMELIER_GRPC_ENDPOINT"),
			expectedResult: 0,
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
			client.SetGRPCEndPoint(tc.endpoint)
			// get status
			status, err := GetStatus(client, tc.chainName)
			if err != nil {
				t.Fatalf("%s: %s", tc.testingName, err.Error())
			}

			for _, validator := range status.Validators {
				if validator.Moniker == testMoniker {
					t.Log("got our test validator information for this package")
					t.Logf("moniker: %s", validator.Moniker)
					t.Logf("valoper address: %s", validator.ValidatorOperatorAddress)
					t.Logf("orchesrator address: %s", validator.OrchestratorAddress)
					t.Logf("eventnonce value: %.f", validator.EventNonce)

					actual := (status.HeighestNonce - validator.EventNonce)
					assert.Equalf(t,
						tc.expectedResult,
						actual,
						"Expected %.f does not match actual %.f", tc.expectedResult, actual,
					)
				}
			}
		})
	}
}
