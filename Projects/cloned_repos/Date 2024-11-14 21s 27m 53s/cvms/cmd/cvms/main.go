package main

import (
	"os"

	"github.com/cosmostation/cvms/cmd"
	"github.com/spf13/cobra"
)

func init() {
	rootCmd.AddCommand(
		cmd.VersionCmd,
		cmd.StartCmd(),
		cmd.ValidateCmd(),
	)
}

func main() {
	err := rootCmd.Execute()
	if err != nil {
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Short: `
 ________  ___      ___ _____ ______   ________      
|\   ____\|\  \    /  /|\   _ \  _   \|\   ____\     
\ \  \___|\ \  \  /  / | \  \\\__\ \  \ \  \___|_    
 \ \  \    \ \  \/  / / \ \  \\|__| \  \ \_____  \   
  \ \  \____\ \    / /   \ \  \    \ \  \|____|\  \  
   \ \_______\ \__/ /     \ \__\    \ \__\____\_\  \ 
    \|_______|\|__|/       \|__|     \|__|\_________\`,
	Use:  "cvms [ start || version ]",
	Args: cobra.NoArgs,
	CompletionOptions: cobra.CompletionOptions{
		DisableDefaultCmd: true,
	},
}
