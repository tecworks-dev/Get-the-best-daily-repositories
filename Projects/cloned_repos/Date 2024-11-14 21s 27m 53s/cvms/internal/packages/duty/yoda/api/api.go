package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/packages/duty/yoda/types"
)

func GetYodaStatus(
	c *common.Exporter,
	CommonYodaQueryPath string,
	CommonYodaParser func([]byte) (float64, error),
) (types.CommonYodaStatus, error) {
	// init context
	ctx, cancel := context.WithTimeout(context.Background(), common.Timeout)
	defer cancel()

	// create requester
	requester := c.APIClient.R().SetContext(ctx)

	// get on-chain validators
	resp, err := requester.Get(types.CommonValidatorQueryPath)
	if err != nil {
		c.Errorf("api error: %s", err)
		return types.CommonYodaStatus{}, common.ErrFailedHttpRequest
	}
	if resp.StatusCode() != http.StatusOK {
		c.Errorf("api error: got %d code from %s", resp.StatusCode(), resp.Request.URL)
		return types.CommonYodaStatus{}, common.ErrGotStrangeStatusCode
	}

	// json unmarsharling received validators data
	var validators types.CommonValidatorsQueryResponse
	if err := json.Unmarshal(resp.Body(), &validators); err != nil {
		c.Errorf("api error: %s", err)
		return types.CommonYodaStatus{}, common.ErrFailedJsonUnmarshal
	}

	// init channel and waitgroup for go-routine
	ch := make(chan helper.Result)
	var wg sync.WaitGroup
	yodaResults := make([]types.ValidatorStatus, 0)

	// add wg by the number of total validators
	wg.Add(len(validators.Validators))

	// get miss count by each validator
	for _, item := range validators.Validators {
		// set query path
		validatorOperatorAddress := item.OperatorAddress
		validatorMoniker := item.Description.Moniker
		queryPath := strings.Replace(CommonYodaQueryPath, "{validator_address}", validatorOperatorAddress, -1)

		// start go-routine
		go func(ch chan helper.Result) {
			defer helper.HandleOutOfNilResponse(c.Entry)
			defer wg.Done()

			resp, err := requester.Get(queryPath)
			if err != nil {
				if resp == nil {
					c.Errorln("[panic] passed resp.Time() nil point err")
					ch <- helper.Result{Item: nil, Success: false}
					return
				}
				c.Errorf("api error: %s", err)
				ch <- helper.Result{Item: nil, Success: false}
				return
			}
			if resp.StatusCode() != 200 {
				c.Errorf("api error: got %d code from %s", resp.StatusCode(), resp.Request.URL)
				ch <- helper.Result{Item: nil, Success: false}
				return
			}

			isActive, err := CommonYodaParser(resp.Body())
			if err != nil {
				c.Errorf("api error: %s", err)
				ch <- helper.Result{Item: nil, Success: false}
				return
			}

			ch <- helper.Result{
				Success: true,
				Item: types.ValidatorStatus{
					IsActive:                 isActive,
					ValidatorOperatorAddress: validatorOperatorAddress,
					Moniker:                  validatorMoniker,
				}}
		}(ch)
		time.Sleep(10 * time.Millisecond)
	}

	// close channel
	go func() {
		wg.Wait()
		close(ch)
	}()

	// collect validator's orch
	errorCount := 0
	for r := range ch {
		if r.Success {
			yodaResults = append(yodaResults, r.Item.(types.ValidatorStatus))
			continue
		}
		errorCount++
	}

	if errorCount > 0 {
		c.Errorf("failed to collect all validator results from node, got errors count: %d", errorCount)
		return types.CommonYodaStatus{}, common.ErrFailedHttpRequest
	}

	return types.CommonYodaStatus{
		Validators: yodaResults,
	}, nil
}
