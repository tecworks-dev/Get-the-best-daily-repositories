package baitroute

import (
	"time"

	"github.com/gofiber/fiber/v2"
)

// RegisterWithFiber registers bait endpoints with a Fiber app
func (b *BaitRoute) RegisterWithFiber(app *fiber.App) error {
	// Register bait endpoints
	for _, rule := range b.rules {
		handler := b.createFiberHandler(rule)

		switch rule.Method {
		case "GET":
			app.Get(rule.Path, handler)
		case "POST":
			app.Post(rule.Path, handler)
		case "PUT":
			app.Put(rule.Path, handler)
		case "DELETE":
			app.Delete(rule.Path, handler)
		case "PATCH":
			app.Patch(rule.Path, handler)
		case "HEAD":
			app.Head(rule.Path, handler)
		case "OPTIONS":
			app.Options(rule.Path, handler)
		}
	}

	return nil
}

// createFiberHandler creates a Fiber-specific handler for an endpoint
func (b *BaitRoute) createFiberHandler(rule Rule) fiber.Handler {
	return func(c *fiber.Ctx) error {
		// Set custom headers
		for key, value := range rule.Headers {
			c.Set(key, value)
		}

		// Set content type
		if rule.ContentType != "" {
			c.Set("Content-Type", rule.ContentType)
		}

		// Trigger alert if handler is set
		if b.alertHandler != nil {
			alert := Alert{
				Method:        c.Method(),
				Path:          c.Path(),
				SourceIP:      c.IP(),
				TrueClientIP:  c.Get("True-Client-IP"),
				XForwardedFor: c.Get("X-Forwarded-For"),
				RuleName:      b.ruleSources[c.Method()+":"+c.Path()],
				Timestamp:     time.Now(),
			}
			go b.alertHandler(alert)
		}

		// Send response with status code and body
		c.Status(rule.Status)
		return c.SendString(rule.Body)
	}
}
