package repository

import (
	"database/sql"
	"log"
	"os"
	"testing"
	"time"

	"github.com/cosmostation/cvms/internal/testutil"
	"github.com/stretchr/testify/assert"
)

var (
	metarepo      *MetaRepository
	TestChainName = "cosmos"
	TestChainID   = "cosmoshub-4"
	IsMainnet     = true
)

func TestMain(m *testing.M) {
	_ = testutil.SetupForTest()
	metarepo = &MetaRepository{10 * time.Second, testutil.TestDB}
	prerunForTest()
	r := m.Run()
	metarepo.Close()
	os.Exit(r)
}

func Test_GetTablesRecords(t *testing.T) {
	_, err := metarepo.CheckIndexpoinerAlreadyInitialized("", 0)
	assert.NoError(t, err)
}

func prerunForTest() {
	isNewChain := false
	chainInfoID, err := metarepo.SelectChainInfoIDByChainID(TestChainID)
	if err != nil {
		if err == sql.ErrNoRows {
			log.Println("this is new chain id")
			isNewChain = true
		} else {
			log.Fatalln(err)
		}
	}

	if isNewChain {
		chainInfoID, err = metarepo.InsertChainInfo(TestChainName, TestChainID, IsMainnet)
		if err != nil {
			log.Fatalln(err)
		}
		log.Printf("%s(%s) chain's chain info id: %d", TestChainName, TestChainID, chainInfoID)
		return
	}

	log.Printf("%s(%s) chain's chain info id: %d", TestChainName, TestChainID, chainInfoID)
}
