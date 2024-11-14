package router

import (
	"os"
	"testing"

	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestYodaStatus(t *testing.T) {

	_ = tests.SetupForTest()
	testMoniker := os.Getenv("TEST_MONIKER")
	TestCases := []struct {
		testingName    string
		chainName      string
		endpoint       string
		expectedResult float64
	}{
		{
			testingName:    "Band Yoda Status Check",
			chainName:      "band",
			endpoint:       os.Getenv("TEST_BAND_ENDPOINT"),
			expectedResult: 1,
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
			assert.NotEmpty(t, status)
			for _, validator := range status.Validators {
				if validator.Moniker == testMoniker {
					actual := validator.IsActive
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
