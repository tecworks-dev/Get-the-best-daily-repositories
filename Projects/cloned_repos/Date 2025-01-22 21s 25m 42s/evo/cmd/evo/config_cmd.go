package main

import (
	"evo/internal/core"
	"fmt"

	"github.com/spf13/cobra"
)

var configCmd = &cobra.Command{
    Use:   "config",
    Short: "Manage Evo configuration",
    Long:  `Use this command to get or set global/local Evo config values.`,
}

var (
    cfgGlobal bool
    cfgLocal  bool
)

var setCmd = &cobra.Command{
    Use:   "set <key> <value>",
    Short: "Set a config key to a value",
    Run: func(cmd *cobra.Command, args []string) {
        if len(args) < 2 {
            fmt.Println("Usage: evo config set [--global|--local] <key> <value>")
            return
        }
        key, val := args[0], args[1]

        if cfgGlobal {
            // global
            gc, _ := core.LoadGlobalConfig()
            gc[key] = val
            if err := core.SaveGlobalConfig(gc); err != nil {
                fmt.Printf("Failed to save global config: %v\n", err)
                return
            }
            fmt.Printf("Set global config %s = %s\n", key, val)
        } else if cfgLocal {
            // local
            repoPath, err := core.FindRepoRoot(".")
            if err != nil {
                fmt.Println("Not in an Evo repository.")
                return
            }
            lc, _ := core.LoadLocalConfig(repoPath)
            lc[key] = val
            if err := core.SaveLocalConfig(repoPath, lc); err != nil {
                fmt.Printf("Failed to save local config: %v\n", err)
                return
            }
            fmt.Printf("Set local config %s = %s\n", key, val)
        } else {
            // else => default to local if neither --global nor --local specified
            repoPath, err := core.FindRepoRoot(".")
            if err != nil {
                fmt.Println("Not in an Evo repository.")
                return
            }
            lc, _ := core.LoadLocalConfig(repoPath)
            lc[key] = val
            if err := core.SaveLocalConfig(repoPath, lc); err != nil {
                fmt.Printf("Failed to save local config: %v\n", err)
                return
            }
            fmt.Printf("Set local config %s = %s\n", key, val)
        }
    },
}

var getCmd = &cobra.Command{
    Use:   "get <key>",
    Short: "Get a config value (local overrides global)",
    Run: func(cmd *cobra.Command, args []string) {
        if len(args) < 1 {
            fmt.Println("Usage: evo config get <key>")
            return
        }
        key := args[0]

        // local override
        repoPath, err := core.FindRepoRoot(".")
        if err == nil {
            lc, _ := core.LoadLocalConfig(repoPath)
            if val, ok := lc[key]; ok {
                fmt.Printf("%s\n", val)
                return
            }
        }

        // fallback to global
        gc, _ := core.LoadGlobalConfig()
        if val, ok := gc[key]; ok {
            fmt.Printf("%s\n", val)
            return
        }
        fmt.Printf("Key '%s' not found in local or global config.\n", key)
    },
}

func init() {
    setCmd.Flags().BoolVar(&cfgGlobal, "global", false, "Use global config (~/.evo/config.json)")
    setCmd.Flags().BoolVar(&cfgLocal, "local", false, "Use repo-level config (.evo/config/local.json)")

    configCmd.AddCommand(setCmd, getCmd)
    rootCmd.AddCommand(configCmd)
}
