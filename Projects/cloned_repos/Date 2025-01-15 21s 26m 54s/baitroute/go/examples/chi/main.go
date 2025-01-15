package main

import (
	"log"
	"net/http"
	"path/filepath"
	"runtime"

	"github.com/go-chi/chi/v5"
	"github.com/utkusen/baitroute/go/pkg/baitroute"
)

func main() {
	// Create a new Chi router
	r := chi.NewRouter()

	// Create a real endpoint
	r.Get("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Welcome to my web application!"))
	})

	// Get the directory of the current file
	_, filename, _, _ := runtime.Caller(0)
	currentDir := filepath.Dir(filename)
	rulesPath := filepath.Join(currentDir, "..", "..", "..", "rules")

	// Initialize baitroute with all rules
	b, err := baitroute.NewBaitRoute(rulesPath)
	if err != nil {
		if err, ok := err.(*baitroute.EndpointConflictError); ok {
			log.Fatalf("Endpoint conflict detected in %s: %s %s is already defined",
				err.SourceFile, err.Method, err.Path)
		}
		log.Fatalf("Failed to initialize baitroute: %v", err)
	}

	/* Alternatively, you can select specific rules to load:
	b, err := baitroute.NewBaitRoute(rulesPath,
		"exposures/aws-credentials",
		"exposures/sql-dump",
		"info/ibm-http-server",
	)
	*/

	// Set up alert handler
	b.OnBaitHit(func(alert baitroute.Alert) {
		// Basic alert logging with core fields
		// Note: Can be integrated with SIEM systems
		log.Printf("Bait Alert - Method: %s, Path: %s, Source IP: %s, Rule: %s",
			alert.Method,
			alert.Path,
			alert.SourceIP,
			alert.RuleName)

		/* Example: Sentry Integration
		sentry.WithScope(func(scope *sentry.Scope) {
			scope.SetExtra("source_ip", alert.SourceIP)
			scope.SetExtra("true_client_ip", alert.TrueClientIP)
			scope.SetExtra("x_forwarded_for", alert.XForwardedFor)
			scope.SetExtra("rule_name", alert.RuleName)
			scope.SetExtra("method", alert.Method)
			scope.SetExtra("path", alert.Path)
			scope.SetTag("event_type", "bait_hit")
			sentry.CaptureMessage("Security Alert: BaitRoute Endpoint Hit")
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

	// Register baitroute endpoints with Chi router
	if err := b.RegisterWithChi(r); err != nil {
		log.Fatalf("Failed to register baitroute endpoints: %v", err)
	}

	// Start the server
	log.Println("Server starting on http://localhost:8087")
	if err := http.ListenAndServe(":8087", r); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
