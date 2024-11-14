package router_test

import (
	"os"
	"testing"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/utility/balance/router"
	tests "github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

func TestRouter(t *testing.T) {
	_ = tests.SetupForTest()

	TestCases := []struct {
		testingName      string
		endpoint         string
		protocolType     string
		balanceAddresses []string
		baseDenom        string
		baseExponent     int
	}{
		{
			testingName:      "Kava Oracle Balance Check",
			protocolType:     "cosmos",
			endpoint:         os.Getenv("TEST_KAVA_ENDPOINT"),
			balanceAddresses: []string{os.Getenv("TEST_KAVA_ADDRESS")},
			baseDenom:        os.Getenv("TEST_KAVA_DENOM"),
			// baseExponent:     os.Getenv("TEST_KAVA_EXPONENT"),
			baseExponent: 6,
		},
		{
			testingName:  "Injective Balances Check",
			endpoint:     os.Getenv("TEST_INJECTIVE_ENDPOINT_1"),
			protocolType: "cosmos",
			balanceAddresses: []string{
				os.Getenv("TEST_INJECTIVE_ADDRESS_1"),
				os.Getenv("TEST_INJECTIVE_ADDRESS_2"),
			},
			baseDenom: os.Getenv("TEST_INJECTIVE_DENOM"),
			// baseExponent: os.Getenv("TEST_INJECTIVE_EXPONENT"),
			baseExponent: 18,
		},
		{
			testingName:  "Ethereum Balances Check",
			endpoint:     os.Getenv("TEST_ETHEREUM_ENDPOINT"),
			protocolType: "ethereum",
			balanceAddresses: []string{
				os.Getenv("TEST_ETHEREUM_ADDRESS_1"),
				os.Getenv("TEST_ETHEREUM_ADDRESS_2"),
				os.Getenv("TEST_ETHEREUM_ADDRESS_3"),
			},
			baseDenom: os.Getenv("TEST_ETHEREUM_DENOM"),
			// baseExponent: os.Getenv("TEST_ETHEREUM_EXPONENT"),
			baseExponent: 18,
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
			p := common.Packager{
				ProtocolType: tc.protocolType,
				OptionPackager: common.OptionPackager{
					BalanceAddresses: tc.balanceAddresses,
					BalanceExponent:  tc.baseExponent,
					BalanceDenom:     tc.baseDenom,
				},
			}

			// get status
			status, err := router.GetStatus(exporter, p)
			assert.Nil(t, err)

			// check & print logs
			for _, item := range status.Balances {
				if !assert.Greater(t, item.RemainingBalance, float64(0), "here are some messages") {
					t.FailNow()
				}

				t.Logf("\naccount: %s | balance: %.2f", item.Address, item.RemainingBalance)
			}

		})
	}
}
