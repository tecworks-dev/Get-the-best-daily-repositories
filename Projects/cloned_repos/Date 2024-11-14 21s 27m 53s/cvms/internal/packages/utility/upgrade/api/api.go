package api

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"strings"

	"github.com/cosmostation/cvms/internal/common"
	"github.com/cosmostation/cvms/internal/packages/utility/upgrade/types"
)

const blockHeightInternal = 1000

func GetUpgradeStatus(
	c *common.Exporter,
	CommonUpgradeQueryPath string, CommonUpgradeParser func([]byte) (int64, string, error),
	CommonBlockQueryPath string, CommonBlockParser func([]byte) (int64, int64, error),
	CommonLatestBlockQueryPath string,
) (types.CommonUpgrade, error) {
	ctx := context.Background()
	ctx, cancel := context.WithTimeout(ctx, common.Timeout)
	defer cancel()

	requester := c.APIClient.R().SetContext(ctx)
	resp, err := requester.Get(CommonUpgradeQueryPath)
	if err != nil {
		c.Errorf("api error: %s", err)
		return types.CommonUpgrade{}, common.ErrFailedHttpRequest
	}
	if resp.StatusCode() != http.StatusOK {
		c.Errorf("api error: status code is %d from %s", resp.StatusCode(), resp.Request.URL)
		return types.CommonUpgrade{}, common.ErrGotStrangeStatusCode
	}

	upgradeHeight, upgradeName, err := CommonUpgradeParser(resp.Body())
	if err != nil {
		c.Errorf("parser error: %s", err)
		return types.CommonUpgrade{}, common.ErrFailedJsonUnmarshal
	}

	// non-exist onchain upgrade
	if upgradeHeight == 0 {
		c.Debugln("nothing to upgrade in on-chain state now")
		return types.CommonUpgrade{}, common.ErrCanSkip
	} else {
		c.Infof("found the onchain upgrade at %d", upgradeHeight)

		// exist onchain upgrade
		resp, err := requester.Get(CommonLatestBlockQueryPath)
		if err != nil {
			c.Errorf("api error: %s", err)
			return types.CommonUpgrade{}, common.ErrFailedHttpRequest
		}
		if resp.StatusCode() != http.StatusOK {
			c.Errorf("api error: [%d] %s", resp.StatusCode(), err)
			return types.CommonUpgrade{}, common.ErrGotStrangeStatusCode
		}

		latestHeight, latestHeightTime, err := CommonBlockParser(resp.Body())
		if err != nil {
			return types.CommonUpgrade{}, err
		}

		previousHeight := (latestHeight - blockHeightInternal)
		previousBlockQueryPath := strings.Replace(CommonBlockQueryPath, "{height}", fmt.Sprint(previousHeight), -1)

		resp, err = requester.Get(previousBlockQueryPath)
		if err != nil {
			c.Errorf("api error: %s", err)
			return types.CommonUpgrade{}, common.ErrFailedHttpRequest
		}
		if resp.StatusCode() != http.StatusOK {
			c.Errorf("api error: [%d] %s", resp.StatusCode(), err)
			return types.CommonUpgrade{}, common.ErrGotStrangeStatusCode
		}

		previousHeight, previousHeightTime, err := CommonBlockParser(resp.Body())
		if err != nil {
			return types.CommonUpgrade{}, err
		}

		// calculate remaining time seconds
		estimatedBlockTime := float64(latestHeightTime-previousHeightTime) / float64(latestHeight-previousHeight)
		rawRemainingTime := float64(upgradeHeight-latestHeight) * estimatedBlockTime
		remainingTime := math.Round(rawRemainingTime)

		c.Infoln("on-chain upgrade's remaining time:", remainingTime, "seconds")
		return types.CommonUpgrade{
			RemainingTime: remainingTime,
			UpgradeName:   upgradeName,
		}, nil
	}
}
