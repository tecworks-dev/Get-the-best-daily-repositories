package baitroute

import "time"

// Rule represents a single bait endpoint configuration
type Rule struct {
	Method      string            `yaml:"method"`
	Path        string            `yaml:"path"`
	Status      int               `yaml:"status"`
	ContentType string            `yaml:"content-type"`
	Headers     map[string]string `yaml:"headers,omitempty"`
	Body        string            `yaml:"body"`
}

// Alert represents a bait endpoint hit event
type Alert struct {
	Method        string
	Path          string
	SourceIP      string
	TrueClientIP  string
	XForwardedFor string
	RuleName      string
	Timestamp     time.Time
	Body          string
}

// AlertHandler is a function that handles bait endpoint hit events
type AlertHandler func(Alert)

// EndpointConflictError is returned when a bait endpoint conflicts with an existing one
type EndpointConflictError struct {
	Method     string
	Path       string
	SourceFile string
}

func (e *EndpointConflictError) Error() string {
	return "endpoint conflict: " + e.Method + " " + e.Path + " is already defined in " + e.SourceFile
}
