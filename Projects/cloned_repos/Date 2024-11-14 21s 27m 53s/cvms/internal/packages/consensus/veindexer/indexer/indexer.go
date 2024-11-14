package indexer

import (
	"database/sql"
	"time"

	"github.com/pkg/errors"

	"github.com/cosmostation/cvms/internal/helper"
	"github.com/cosmostation/cvms/internal/helper/healthcheck"

	"github.com/cosmostation/cvms/internal/common"
	indexertypes "github.com/cosmostation/cvms/internal/common/indexer/types"
	"github.com/cosmostation/cvms/internal/packages/consensus/veindexer/repository"
)

var (
	supportedProtocolTypes = []string{"cosmos"}
	subsystem              = "extension_vote"
)

type VEIndexer struct {
	*common.Indexer
	repo repository.VEIndexerRepository
}

// Compile-time Assertion
var _ common.IIndexer = (*VEIndexer)(nil)

// NOTE: this is for solo mode
func NewVEIndexer(p common.Packager) (*VEIndexer, error) {
	status := helper.GetOnChainStatus(p.RPCs, p.ProtocolType)
	if status.ChainID == "" {
		return nil, errors.New("failed to create new veindexer")
	}
	indexer := common.NewIndexer(p, p.Package, status.ChainID)
	repo := repository.NewRepository(*p.IndexerDB, indexertypes.SQLQueryMaxDuration)
	indexer.Lh = indexertypes.LatestHeightCache{LatestHeight: status.BlockHeight}
	return &VEIndexer{indexer, repo}, nil
}

func (veidx *VEIndexer) Start() error {
	if ok := helper.Contains(supportedProtocolTypes, veidx.ProtocolType); ok {

		err := veidx.InitChainInfoID()
		if err != nil {
			return errors.Wrap(err, "failed to init chain_info_id")
		}

		alreadyInit, err := veidx.repo.CheckIndexpoinerAlreadyInitialized(repository.IndexName, veidx.ChainInfoID)
		if err != nil {
			return errors.Wrap(err, "failed to check init tables")
		}
		if !alreadyInit {
			veidx.Warnln("it's not initialized in the database, so that veindexer will init for this package")
			veidx.repo.InitPartitionTablesByChainInfoID(repository.IndexName, veidx.ChainID, veidx.Lh.LatestHeight)
		}

		// NOTE:  ...
		maxBackOffCnt := 5
		cnt := 0
	retryLoop:
		// get last index pointer, index pointer is always initalize if not exist
		initIndexPointer, err := veidx.repo.GetLastIndexPointerByIndexTableName(repository.IndexName, veidx.ChainInfoID)
		if err != nil {
			if cnt < maxBackOffCnt {
				veidx.repo.InitPartitionTablesByChainInfoID(repository.IndexName, veidx.ChainID, veidx.Lh.LatestHeight)
				cnt++
				veidx.Warnln("found unexpected init index pointer and so retry until max 5 times")
				goto retryLoop
			}
			return errors.Wrap(err, "failed to get last index pointer")
		}

		err = veidx.FetchValidatorInfoList()
		if err != nil {
			return errors.Wrap(err, "failed to fetch validator_info list")
		}

		veidx.Infof("loaded index pointer(last saved height): %d", initIndexPointer.Pointer)
		veidx.Infof("initial vim length: %d for %s chain", len(veidx.Vim), veidx.ChainID)

		// init indexer metrics
		veidx.initLabelsAndMetrics()
		// go fetch new height in loop, it must be after init metrics
		go veidx.FetchLatestHeight()
		// loop
		go veidx.Loop(initIndexPointer.Pointer)
		// loop update recent miss counter metrics
		go func() {
			for {
				veidx.Infoln("update recent miss counter metrics and sleep 5s sec...")
				veidx.updateRecentMissCounterMetric()
				time.Sleep(time.Second * 5)
			}
		}()
		// loop partion table time retention by env parameter
		go func() {
			for {
				veidx.Infof("for time retention, delete old records over %s and sleep %s", veidx.RetentionPeriod, indexertypes.RetentionQuerySleepDuration)
				veidx.repo.DeleteOldValidatorExtensionVoteList(veidx.ChainID, veidx.RetentionPeriod)
				time.Sleep(indexertypes.RetentionQuerySleepDuration)
			}
		}()
		return nil
	}

	return nil
}

func (veidx *VEIndexer) Loop(indexPoint int64) {
	isUnhealth := false
	for {
		// node health check
		if isUnhealth {
			healthAPIs := healthcheck.FilterHealthEndpoints(veidx.APIs, veidx.ProtocolType)
			for _, api := range healthAPIs {
				veidx.SetAPIEndPoint(api)
				veidx.Warnf("API endpoint will be changed with health endpoint for this package: %s", api)
				isUnhealth = false
				break
			}

			healthRPCs := healthcheck.FilterHealthRPCEndpoints(veidx.RPCs, veidx.ProtocolType)
			for _, rpc := range healthRPCs {
				veidx.SetRPCEndPoint(rpc)
				veidx.Warnf("RPC endpoint will be changed with health endpoint for this package: %s", rpc)
				isUnhealth = false
				break
			}

			if len(healthAPIs) == 0 || len(healthRPCs) == 0 {
				isUnhealth = true
				veidx.Errorln("failed to get any health endpoints from healthcheck filter, retry sleep 10s")
				time.Sleep(indexertypes.UnHealthSleep)
				continue
			}
		}

		// set new index point height
		newIndexPointerHeight := indexPoint + 1

		// trying to sync with new index pointer height
		newIndexPointer, err := veidx.batchSync(indexPoint, newIndexPointerHeight)
		if err != nil {
			common.Health.With(veidx.RootLabels).Set(0)
			common.Ops.With(veidx.RootLabels).Inc()
			isUnhealth = true
			veidx.Errorf("failed to sync validators vote status in %d height: %s\nit will be retried after sleep %s...",
				indexPoint, err, indexertypes.AfterFailedRetryTimeout.String(),
			)
			time.Sleep(indexertypes.AfterFailedRetryTimeout)
			continue
		}

		// update index point
		indexPoint = newIndexPointer

		// update health and ops
		common.Health.With(veidx.RootLabels).Set(1)
		common.Ops.With(veidx.RootLabels).Inc()

		// logging & sleep
		if veidx.Lh.LatestHeight > indexPoint {
			// when node catching_up is true, sleep 100 milli sec
			veidx.WithField("catching_up", true).
				Infof("latest height is %d but updated index pointer is %d ... remaining %d blocks", veidx.Lh.LatestHeight, indexPoint, (veidx.Lh.LatestHeight - indexPoint))
			time.Sleep(indexertypes.CatchingUpSleepDuration)
		} else {
			// when node already catched up, sleep 5 sec
			veidx.WithField("catching_up", false).
				Infof("updated index pointer to %d and sleep %s sec...", indexPoint, indexertypes.DefaultSleepDuration.String())
			time.Sleep(indexertypes.DefaultSleepDuration)
		}
	}
}

// insert chain-info into chain_info table
func (veidx *VEIndexer) InitChainInfoID() error {
	isNewChain := false
	var chainInfoID int64
	chainInfoID, err := veidx.repo.SelectChainInfoIDByChainID(veidx.ChainID)
	if err != nil {
		if err == sql.ErrNoRows {
			veidx.Infof("this is new chain id: %s", veidx.ChainID)
			isNewChain = true
		} else {
			return errors.Wrap(err, "failed to select chain_info_id by chain-id")
		}
	}

	if isNewChain {
		chainInfoID, err = veidx.repo.InsertChainInfo(veidx.ChainName, veidx.ChainID, veidx.Mainnet)
		if err != nil {
			return errors.Wrap(err, "failed to insert new chain_info_id by chain-id")
		}
	}

	veidx.ChainInfoID = chainInfoID
	return nil
}

func (veidx *VEIndexer) FetchValidatorInfoList() error {
	// get already saved validator-set list for mapping validators ids
	validatorInfoList, err := veidx.repo.GetValidatorInfoListByChainInfoID(veidx.ChainInfoID)
	if err != nil {
		return errors.Wrap(err, "failed to get validator info list")
	}

	// when the this pacakge starts, set validator-id map
	for _, validator := range validatorInfoList {
		veidx.Vim[validator.HexAddress] = int64(validator.ID)
	}

	return nil
}
