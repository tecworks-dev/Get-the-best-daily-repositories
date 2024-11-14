package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var (
	AppName = ""
	Version = ""
	Commit  = ""
)

var VersionCmd = &cobra.Command{
	Use:   "version",
	Short: "Show information about the current binary build",
	Args:  cobra.NoArgs,
	Run:   printBuildInfo,
}

func printBuildInfo(_ *cobra.Command, _ []string) {
	fmt.Printf("AppName : %s\n", AppName)
	fmt.Printf("Version : %s\n", Version)
	fmt.Printf("Commit  : %s\n", Commit)
}
