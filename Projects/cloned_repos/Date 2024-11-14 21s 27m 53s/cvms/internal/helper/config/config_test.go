package config

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

var (
	cfgPath string
)

func TestMain(m *testing.M) {
	// setup
	cmd := exec.Command("go", "env", "GOMOD")
	out, _ := cmd.Output()
	rootPath := strings.Split(string(out), "/go.mod")[0]

	// load
	cfgPath = fmt.Sprintf("%s/%s", rootPath, "config.yaml")

	// exit
	os.Exit(m.Run())

}

func TestParseConfigYaml(t *testing.T) {
	cfg, err := GetConfig(cfgPath)
	if err != nil {
		t.Logf("got error: %s", err)
		t.FailNow()
	}

	_ = cfg
	t.Log("the config.yaml file was parsed successfully")
}

func TestGetConfig(t *testing.T) {
	cfg, err := GetConfig(cfgPath)
	assert.NoError(t, err)

	supportChains, err := GetSupportChainConfig()
	assert.NoError(t, err)

	skipCnt := 0
	for _, cc := range cfg.ChainConfigs {
		// support chains
		chainDetails, exists := supportChains.Chains[cc.ChainID]
		if exists {
			fmt.Printf("\tChain Config for '%s'\n", cc.ChainID)
			fmt.Printf("\t\tSupport ChainName is %s\n", chainDetails.ChainName)
			fmt.Printf("\t\tSupport Mainnet Flag is %v\n", chainDetails.Mainnet)
			fmt.Printf("\t\tSupport Protocol is %s\n", chainDetails.ProtocolType)
			fmt.Printf("\t\tSupport Asset are %v\n", chainDetails.SupportAsset)
			fmt.Printf("\t\tSupport Packages are %v\n", chainDetails.Packages)
		} else {
			fmt.Printf("No details found for chain: %s\nPlease visit chainlist repository, make a PR to support your chain", cc.DisplayName)
			t.SkipNow()
			skipCnt++
		}

		fmt.Println()
		// nodes
		for idx, node := range cc.Nodes {
			fmt.Printf("\t\tIndex(%d) Node Endpoints\n", idx)
			fmt.Printf("\t\tRPC: %s\tAPI: %s\tGRPC: %s\n", node.RPC, node.API, node.GRPC)
		}

		fmt.Println()
	}

	if skipCnt > 0 {
		t.Log("Check your getconfig function")
	}
}

func TestGetConfigForConsumerChains(t *testing.T) {
	cfg, err := GetConfig(cfgPath)
	assert.NoError(t, err)

	supportChains, err := GetSupportChainConfig()
	assert.NoError(t, err)

	skipCnt := 0
	for _, cc := range cfg.ChainConfigs {
		// support chains
		chainDetails, exists := supportChains.Chains[cc.ChainID]
		if chainDetails.Consumer {
			if exists {
				fmt.Printf("\tChain Config for '%s'\n", cc.ChainID)
				fmt.Printf("\t\tSupport ChainName is %s\n", chainDetails.ChainName)
				fmt.Printf("\t\tSupport Mainnet Flag is %v\n", chainDetails.Mainnet)
				fmt.Printf("\t\tSupport Protocol is %s\n", chainDetails.ProtocolType)
				fmt.Printf("\t\tSupport Asset are %v\n", chainDetails.SupportAsset)
				fmt.Printf("\t\tSupport Packages are %v\n", chainDetails.Packages)
			} else {
				fmt.Printf("No details found for chain: %s\nPlease visit chainlist repository, make a PR to support your chain", cc.DisplayName)
				t.SkipNow()
				skipCnt++
			}

			fmt.Println()
			// nodes
			for idx, node := range cc.Nodes {
				fmt.Printf("\t\tIndex(%d) Node Endpoints\n", idx)
				fmt.Printf("\t\tRPC: %s\tAPI: %s\tGRPC: %s\n", node.RPC, node.API, node.GRPC)
			}

			// nodes
			for idx, node := range cc.ProviderNodes {
				fmt.Printf("\t\tIndex(%d) Provider Node Endpoints\n", idx)
				fmt.Printf("\t\tRPC: %s\tAPI: %s\tGRPC: %s\n", node.RPC, node.API, node.GRPC)
			}

			fmt.Println()
		}

	}

	if skipCnt > 0 {
		t.Log("Check your getconfig function")
	}
}

func TestGetSupportChainConfig(t *testing.T) {
	supportChains, err := GetSupportChainConfig()
	assert.NoError(t, err)

	for chainID, chainDetail := range supportChains.Chains {
		fmt.Printf("Support Chain Detail: %s - %v\n", chainID, chainDetail)
	}
}
func TestGetSupportChainConfigForConsumerChains(t *testing.T) {
	supportChains, err := GetSupportChainConfig()
	assert.NoError(t, err)
	for chainID, chainDetail := range supportChains.Chains {
		if chainDetail.Consumer {
			fmt.Printf("Support Chain Detail: %s - %v\n", chainID, chainDetail)
		}
	}
}
