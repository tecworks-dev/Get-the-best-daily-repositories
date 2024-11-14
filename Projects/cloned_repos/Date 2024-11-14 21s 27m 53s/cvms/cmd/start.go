package cmd

import (
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"github.com/cosmostation/cvms/internal/app/exporter"
	"github.com/cosmostation/cvms/internal/app/indexer"
	"github.com/cosmostation/cvms/internal/helper/config"
	"github.com/cosmostation/cvms/internal/helper/logger"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
)

var flagSets []*pflag.FlagSet

func init() {
	flagSets = []*pflag.FlagSet{
		ConfigFlag(),
		LogFlags(),
		PortFlag(),
		FilterFlag(),
	}
}

var setFlags = func(cmd *cobra.Command) {
	for _, set := range flagSets {
		cmd.Flags().AddFlagSet(set)
	}
}

func StartCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:  "start",
		Args: cobra.ExactArgs(1),
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println("CVMS Start subcommands")
		},
	}
	cmd.AddCommand(StartIndexerCmd(), StartExporterCmd())
	return cmd
}

func StartExporterCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "exporter",
		Short: "Start CVMS Exporter!",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			ctx := cmd.Context()
			logLevel := cmd.Flag(LogLevel).Value.String()
			logColorDisable := cmd.Flag(LogColorDisable).Value.String()
			configfile := cmd.Flag(Config).Value.String()
			port := cmd.Flag(Port).Value.String()

			cfg, err := config.GetConfig(configfile)
			if err != nil {
				return err
			}

			supportChains, err := config.GetSupportChainConfig()
			if err != nil {
				return err
			}

			logger, err := logger.GetLogger(logColorDisable, logLevel)
			if err != nil {
				return err
			}

			exporterServer, err := exporter.Build(port, logger, cfg, supportChains)
			if err != nil {
				return err
			}

			sigs := make(chan os.Signal, 1)
			signal.Notify(sigs, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
			done := make(chan struct{})
			go func() {
				<-sigs
				logger.Println("Received interrupt signal, shutting down...")
				if err := exporterServer.Shutdown(ctx); err != nil {
					logger.Fatalf("Server Shutdown Failed:%+v", err)
				}
				logger.Println("Server Stopped")
				close(done)
			}()

			if err := exporterServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("Listen error: %v", err)
			}

			<-done
			logger.Println("Server Exited Properly")
			return nil
		},
	}
	setFlags(cmd)
	return cmd
}

func StartIndexerCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "indexer",
		Short: "Start CVMS Indexer!",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) (err error) {
			ctx := cmd.Context()
			logLevel := cmd.Flag(LogLevel).Value.String()
			logColorDisable := cmd.Flag(LogColorDisable).Value.String()
			configfile := cmd.Flag(Config).Value.String()
			port := cmd.Flag(Port).Value.String()

			cfg, err := config.GetConfig(configfile)
			if err != nil {
				return err
			}

			supportChains, err := config.GetSupportChainConfig()
			if err != nil {
				return err
			}

			logger, err := logger.GetLogger(logColorDisable, logLevel)
			if err != nil {
				return err
			}

			indexerServer, err := indexer.Build(port, logger, cfg, supportChains)
			if err != nil {
				return err
			}

			sigs := make(chan os.Signal, 1)
			signal.Notify(sigs, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)
			done := make(chan struct{})
			go func() {
				<-sigs
				logger.Println("Received interrupt signal, shutting down...")
				if err := indexerServer.Shutdown(ctx); err != nil {
					logger.Fatalf("Server Shutdown Failed:%+v", err)
				}
				logger.Println("Server Stopped")
				close(done)
			}()

			if err := indexerServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
				logger.Fatalf("Listen error: %v", err)
			}

			<-done
			logger.Println("Server Exited Properly")
			return nil
		},
	}
	setFlags(cmd)
	return cmd
}
