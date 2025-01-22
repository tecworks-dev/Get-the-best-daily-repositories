package core

import (
	"encoding/json"
	"os"
	"path/filepath"
)

func LoadGlobalConfig() (map[string]string, error) {
	m := make(map[string]string)
	home, err := os.UserHomeDir()
	if err != nil {
		return m, err
	}
	gcPath := filepath.Join(home, ".evo", "config.json")
	b, err := os.ReadFile(gcPath)
	if os.IsNotExist(err) {
		return m, nil
	} else if err != nil {
		return m, err
	}
	json.Unmarshal(b, &m)
	return m, nil
}

func SaveGlobalConfig(cfg map[string]string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	gcPath := filepath.Join(home, ".evo", "config.json")
	b, _ := json.MarshalIndent(cfg, "", "  ")
	return os.WriteFile(gcPath, b, 0644)
}

func LoadLocalConfig(repoPath string) (map[string]string, error) {
	cfgPath := filepath.Join(repoPath, EvoDir, "config", "local.json")
	m := make(map[string]string)
	b, err := os.ReadFile(cfgPath)
	if os.IsNotExist(err) {
		return m, nil // empty
	} else if err != nil {
		return m, err
	}
	json.Unmarshal(b, &m)
	return m, nil
}

func SaveLocalConfig(repoPath string, m map[string]string) error {
	cfgPath := filepath.Join(repoPath, EvoDir, "config", "local.json")
	b, _ := json.MarshalIndent(m, "", "  ")
	return os.WriteFile(cfgPath, b, 0644)
}
