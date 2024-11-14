package cmd

import (
	"github.com/cosmostation/cvms/internal/helper/config"
	"github.com/spf13/cobra"
)

func ValidateCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "validate",
		Short: "Validate current config.yaml before starting application",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			ctx := cmd.Context()
			_ = ctx
			configfile := cmd.Flag(Config).Value.String()

			cfg, err := config.GetConfig(configfile)
			if err != nil {
				return err
			}

			cmd.Println("Your config.yaml file would be parsed successfully")
			cmd.Printf("\tMoniker : %v\n", cfg.Monikers)
			cmd.Printf("\tChains : %d\n", len(cfg.ChainConfigs))
			return nil
		},
	}
	setFlags(cmd)
	return cmd
}
