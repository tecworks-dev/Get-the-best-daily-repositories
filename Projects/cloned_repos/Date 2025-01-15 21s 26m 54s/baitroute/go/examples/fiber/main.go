package main

import (
	"log"
	"path/filepath"
	"runtime"

	"github.com/gofiber/fiber/v2"
	"github.com/utkusen/baitroute/go/pkg/baitroute"
)

func main() {
	// Create a new Fiber app
	app := fiber.New()

	// Create a real endpoint
	app.Get("/", func(c *fiber.Ctx) error {
		return c.SendString("Welcome to my web application!")
	})

	// Get the directory of the current file
	_, filename, _, _ := runtime.Caller(0)
	currentDir := filepath.Dir(filename)
	rulesPath := filepath.Join(currentDir, "..", "..", "..", "rules")

	// Initialize baitroute with all rules (default behavior)
	b, err := baitroute.NewBaitRoute(rulesPath)
	if err != nil {
		if err, ok := err.(*baitroute.EndpointConflictError); ok {
			log.Fatalf("Endpoint conflict detected in %s: %s %s is already defined",
				err.SourceFile, err.Method, err.Path)
		}
		log.Fatalf("Failed to initialize baitroute: %v", err)
	}

	/* Alternatively, you can select specific rules to load:
	b, err := baitroute.NewBaitRoute("../../../rules",
		"exposures/aws-credentials",
		"exposures/sql-dump",
		"info/ibm-http-server",
	)
	*/

	// Set up alert handler for bait hits
	b.OnBaitHit(func(alert baitroute.Alert) {
		// Log the alert details
		log.Printf("[SECURITY ALERT] Bait endpoint hit: %s %s\n", alert.Method, alert.Path)
		log.Printf("Source IP: %s\n", alert.SourceIP)
		if alert.TrueClientIP != "" {
			log.Printf("True-Client-IP: %s\n", alert.TrueClientIP)
		}
		if alert.XForwardedFor != "" {
			log.Printf("X-Forwarded-For: %s\n", alert.XForwardedFor)
		}
		log.Printf("Rule: %s\n", alert.RuleName)

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

	// Register bait endpoints with Fiber app
	if err := b.RegisterWithFiber(app); err != nil {
		log.Fatalf("Failed to register bait endpoints: %v", err)
	}

	// Start the server
	log.Println("Server starting on http://localhost:8087")
	if err := app.Listen(":8087"); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}
