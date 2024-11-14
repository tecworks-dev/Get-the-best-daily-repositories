package config

import (
	"os"

	"github.com/pkg/errors"
	"gopkg.in/yaml.v3"
)

type SupportChains struct {
	Chains map[string]ChainDetail `yaml:",inline"`
}

type ChainDetail struct {
	ChainName    string   `yaml:"chain_name"`
	ProtocolType string   `yaml:"protocol_type"`
	Mainnet      bool     `yaml:"mainnet"`
	Consumer     bool     `yaml:"consumer"`
	Packages     []string `yaml:"packages"`
	SupportAsset Asset    `yaml:"support_asset"`
}

type Asset struct {
	Denom   string `yaml:"denom"`
	Decimal int    `yaml:"decimal"`
}

func GetSupportChainConfig() (*SupportChains, error) {
	dataBytes, err := os.ReadFile(MustGetSupportChainPath("support_chains.yaml"))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read config file")
	}

	ctDataBytes, err := os.ReadFile(MustGetSupportChainPath("custom_chains.yaml"))
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read config file")
	}

	scCfg := &SupportChains{}
	err = yaml.Unmarshal(dataBytes, scCfg)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode config file")
	}

	ctCfg := &SupportChains{}
	err = yaml.Unmarshal(ctDataBytes, ctCfg)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to decode second config file")
	}

	// Merge the two configurations
	for chainName, chainDetail := range ctCfg.Chains {
		if _, exists := scCfg.Chains[chainName]; exists {
			return nil, errors.Errorf("duplicate chain found: %s", chainName)
		}

		// Add custom chains by custom_chains.yaml
		scCfg.Chains[chainName] = chainDetail
	}
	return scCfg, nil
}
