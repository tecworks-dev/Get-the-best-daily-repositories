package core

import (
	"encoding/json"
	"os"
	"path/filepath"
)


type UserConfig struct {
    Name  string `json:"name"`
    Email string `json:"email"`
}

// LoadUserConfig from .evo/config/user.json
func LoadUserConfig(repoPath string) (UserConfig, error) {
    var uc UserConfig
    p := filepath.Join(repoPath, EvoDir, "config", "user.json")
    b, err := os.ReadFile(p)
    if err != nil {
        return uc, err
    }
    err = json.Unmarshal(b, &uc)
    return uc, err
}

// SaveUserConfig if needed
func SaveUserConfig(repoPath string, uc UserConfig) error {
    p := filepath.Join(repoPath, EvoDir, "config")
    os.MkdirAll(p, 0755)
    data, _ := json.MarshalIndent(uc, "", "  ")
    return os.WriteFile(filepath.Join(p, "user.json"), data, 0644)
}
