package repository

import (
	"testing"
	"time"

	"github.com/cosmostation/cvms/internal/testutil"
)

func Test_SelectRecentValidatorExtensionVoteList(t *testing.T) {
	_ = testutil.SetupForTest()
	repo := NewRepository(testutil.TestIndexerDB, 10*time.Second)
	list, err := repo.SelectRecentValidatorExtensionVoteList("dydx_mainnet_1")
	if err != nil {
		t.Logf("unexpeced err: %s", err)
	}
	t.Log(list)
}
