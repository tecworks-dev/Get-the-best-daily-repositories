package main

import (
	"log"
	"path/filepath"
	"runtime"

	"github.com/utkusen/baitroute/go/pkg/baitroute"
	"github.com/valyala/fasthttp"
)

func main() {
	// Get the directory of the current file
	_, filename, _, _ := runtime.Caller(0)
	currentDir := filepath.Dir(filename)
	rulesPath := filepath.Join(currentDir, "..", "..", "..", "rules")

	// Initialize baitroute with default rules
	b, err := baitroute.NewBaitRoute(rulesPath)
	if err != nil {
		log.Fatalf("Failed to initialize baitroute: %v", err)
	}

	/* Alternatively, you can select specific rules to load:
	b, err := baitroute.NewBaitRoute(rulesPath,
		"exposures/aws-credentials",
		"exposures/sql-dump",
		"info/ibm-http-server",
	)
	*/

	// Configure alert handler
	b.OnBaitHit(func(alert baitroute.Alert) {
		// Basic alert logging for SIEM integration
		// SIEM Integration Note: Format alerts according to your SIEM system requirements
		log.Printf("[ALERT] Bait triggered - Method=%s Path=%s SourceIP=%s Rule=%s",
			alert.Method,
			alert.Path,
			alert.SourceIP,
			alert.RuleName)

		/* Example: Sentry Integration
		sentry.WithScope(func(scope *sentry.Scope) {
			scope.SetLevel(sentry.LevelWarning)
			scope.SetExtra("source_ip", alert.SourceIP)
			scope.SetExtra("true_client_ip", alert.TrueClientIP)
			scope.SetExtra("x_forwarded_for", alert.XForwardedFor)
			scope.SetExtra("rule_name", alert.RuleName)
			scope.SetExtra("method", alert.Method)
			scope.SetExtra("path", alert.Path)
			scope.SetTag("event_type", "bait_hit")
			sentry.CaptureMessage("Security Alert: Bait Endpoint Hit")
		})

		Example: Splunk Integration
		splunk.Send(&splunk.Event{
			Source:    "honeypot",
			Event:     "bait_hit",
			Severity:  "warning",
			SourceIP:  alert.SourceIP,
			ClientIP:  alert.TrueClientIP,
			X-Forwarded-For: alert.XForwardedFor,
			Method:    alert.Method,
			Path:      alert.Path,
			RuleName:  alert.RuleName,
			Timestamp: alert.Timestamp,
		})

		*/
	})

	// Get the FastHTTP request handler
	baitHandler := b.RegisterWithFastHTTP()

	// Create a router function that combines real and bait endpoints
	router := func(ctx *fasthttp.RequestCtx) {
		// Try bait handler first
		baitHandler(ctx)

		// Handle real endpoints if path is not handled by bait
		if string(ctx.Path()) == "/" {
			ctx.WriteString("Welcome to my web application!")
			return
		}

		// Handle 404 Not Found if not already handled
		if ctx.Response.StatusCode() == 0 {
			ctx.Error("Not Found", fasthttp.StatusNotFound)
		}
	}

	// Start the server
	log.Println("Server starting on http://localhost:8087")
	if err := fasthttp.ListenAndServe(":8087", router); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
