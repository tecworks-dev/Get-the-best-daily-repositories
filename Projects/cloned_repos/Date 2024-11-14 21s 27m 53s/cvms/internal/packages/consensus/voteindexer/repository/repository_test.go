package repository

import (
	"testing"
	"time"

	"github.com/cosmostation/cvms/internal/testutil"
)

var (
	TestChainName = "cosmos"
	TestChainID   = "cosmoshub-4"
)

func TestXxx(t *testing.T) {
	_ = testutil.SetupForTest()
	repo := NewRepository(testutil.TestIndexerDB, 10*time.Second)
	list, err := repo.SelectRecentMissValidatorVoteList("althea_258432_1")
	if err != nil {
		t.Logf("unexpeced err: %s", err)
	}
	t.Log(list)
}

// func prerunForTest() {
// 	isNewChain := false
// 	chainInfoID, err := repo.SelectChainInfoIDByChainID(TestChainID)
// 	if err != nil {
// 		if err == sql.ErrNoRows {
// 			log.Println("this is new chain id")
// 			isNewChain = true
// 		} else {
// 			log.Fatalln(err)
// 		}
// 	}

// 	if isNewChain {
// 		chainInfoID, err = repo.InsertChainInfo(TestChainName, TestChainID)
// 		if err != nil {
// 			log.Fatalln(err)
// 		}
// 		log.Printf("%s(%s) chain's chain info id: %d", TestChainName, TestChainID, chainInfoID)
// 		return
// 	}

// 	log.Printf("%s(%s) chain's chain info id: %d", TestChainName, TestChainID, chainInfoID)
// }

// // NOTE: it should be passed
// func Test_CreatePartitionTables(t *testing.T) {
// 	err := repo.CreateValidatorInfoPartitionTableByChainID(TestChainID)
// 	assert.NoError(t, err)

// 	err = repo.CreateVoteIndexerPartitionTableByChainID(TestChainID)
// 	assert.NoError(t, err)
// }

// func Test_InitializeIndexPointer(t *testing.T) {
// 	err := repo.InitializeIndexPointerByChainID(TestChainID, 10000)
// 	assert.NoError(t, err)
// }

// func Test_InsertValidatorVoteList(t *testing.T) {
// 	chainInfoID, err := repo.SelectChainInfoIDByChainID(TestChainID)
// 	assert.NoError(t, err)

// 	// it will be made from makeValidatorVote function
// 	ValidatorVoteList := []model.ValidatorVote{
// 		{
// 			ChainInfoID:           chainInfoID, // mocha-4
// 			ValidatorHexAddressID: 2,
// 			Height:                767173,
// 			Timestamp:             time.Unix(1703140759, 0),
// 		},
// 		{
// 			ChainInfoID:           chainInfoID, // mocha-4
// 			ValidatorHexAddressID: 1,
// 			Height:                767173,
// 			Timestamp:             time.Unix(1703140759, 0),
// 		},
// 	}

// 	err = repo.InsertValidatorVoteList(chainInfoID, 767173, ValidatorVoteList)
// 	assert.NoError(t, err)
// }

// func Test_InsertValidatorInfoList(t *testing.T) {
// 	chainInfoID, err := repo.SelectChainInfoIDByChainID(TestChainID)
// 	assert.NoError(t, err)

// 	validatorInfoList := []common.ValidatorInfo{
// 		{
// 			ChainInfoID: chainInfoID,
// 			HexAddress:  "7619BFC85B72E319BF414A784D4DE40EE9B92C16",
// 		},
// 		{
// 			ChainInfoID: chainInfoID,
// 			HexAddress:  "762CBA617226A799D898F134DD12661C7F1129EB",
// 		},
// 	}

// 	err = repo.InsertValidatorInfoList(validatorInfoList)
// 	assert.NoError(t, err)
// }

// func Test_GetValidatorInfoListByMonikers(t *testing.T) {
// 	chainInfoID, err := repo.SelectChainInfoIDByChainID(TestChainID)
// 	assert.NoError(t, err)

// 	monikers := strings.Split(os.Getenv("TEST_MONIKERS"), ",")
// 	t.Log(monikers)

// 	validatorInfoList, err := repo.GetValidatorInfoListByMonikers(chainInfoID, monikers)
// 	assert.NoError(t, err)

// 	for _, vv := range validatorInfoList {
// 		t.Logf("%s", vv.String())
// 	}
// }

// func Test_TimeRetention(t *testing.T) {
// 	affectedRows, err := repo.DeleteOldValidatorVoteList(TestChainID, "1h")
// 	assert.NoError(t, err)
// 	t.Logf("%d rows deleted\n", affectedRows)
// }
