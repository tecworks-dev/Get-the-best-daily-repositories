package config

import (
	"encoding/json"
	"log"
	"strings"

	"github.com/angelicmuklu/saf/internal/infra/cmq"
	"github.com/angelicmuklu/saf/internal/infra/logger"
	"github.com/angelicmuklu/saf/internal/infra/output"
	"github.com/angelicmuklu/saf/internal/infra/telemetry"
	"github.com/knadh/koanf/parsers/toml"
	"github.com/knadh/koanf/providers/env"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/providers/structs"
	"github.com/knadh/koanf/v2"
	"github.com/tidwall/pretty"
	"go.uber.org/fx"
)

const (
	// prefix indicates environment variables prefix.
	prefix = "saf_"
)

// Config holds all configurations.
type Config struct {
	fx.Out

	Logger    logger.Config    `json:"logger,omitempty"    koanf:"logger"`
	Telemetry telemetry.Config `json:"telemetry,omitempty" koanf:"telemetry"`
	NATS      cmq.Config       `json:"nats,omitempty"      koanf:"nats"`
	Channels  output.Config    `json:"channels,omitempty"  koanf:"channels"`
}

func Provide() Config {
	k := koanf.New(".")

	// load default configuration from default function
	if err := k.Load(structs.Provider(Default(), "koanf"), nil); err != nil {
		log.Fatalf("error loading default: %s", err)
	}

	// load configuration from file
	if err := k.Load(file.Provider("config.toml"), toml.Parser()); err != nil {
		log.Printf("error loading config.toml: %s", err)
	}

	// load environment variables
	if err := k.Load(
		// replace __ with . in environment variables so you can reference field a in struct b
		// as a__b.
		env.Provider(prefix, ".", func(source string) string {
			base := strings.ToLower(strings.TrimPrefix(source, prefix))

			return strings.ReplaceAll(base, "__", ".")
		}),
		nil,
	); err != nil {
		log.Printf("error loading environment variables: %s", err)
	}

	var instance Config
	if err := k.Unmarshal("", &instance); err != nil {
		log.Fatalf("error unmarshalling config: %s", err)
	}

	indent, err := json.MarshalIndent(instance, "", "\t")
	if err != nil {
		panic(err)
	}

	indent = pretty.Color(indent, nil)

	log.Printf(`
================ Loaded Configuration ================
%s
======================================================
	`, string(indent))

	return instance
}
