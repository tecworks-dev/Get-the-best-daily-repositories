package common

import (
	"time"
)

const (
	retryCount               = 10
	retryWaitTimeDuration    = 10 * time.Millisecond
	retryMaxWaitTimeDuration = 3 * time.Second
)

type Exporter struct {
	CommonApp
	Monikers  []string
	ChainName string
	ChainID   string
}

func NewExporter(p Packager) *Exporter {
	app := NewCommonApp(p)
	// NOTE: empty monikers mean all mode
	monikers := []string{}
	if p.Mode == VALIDATOR {
		monikers = p.Monikers
	}
	return &Exporter{
		app,
		monikers,
		p.ChainName,
		p.ChainID,
	}
}
