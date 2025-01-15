package baitroute

import (
	"time"

	"github.com/labstack/echo/v4"
)

// RegisterWithEcho registers bait endpoints with an Echo router
func (b *BaitRoute) RegisterWithEcho(e *echo.Echo) error {
	// Register bait endpoints
	for _, rule := range b.rules {
		handler := b.createEchoHandler(rule)

		switch rule.Method {
		case "GET":
			e.GET(rule.Path, handler)
		case "POST":
			e.POST(rule.Path, handler)
		case "PUT":
			e.PUT(rule.Path, handler)
		case "DELETE":
			e.DELETE(rule.Path, handler)
		case "PATCH":
			e.PATCH(rule.Path, handler)
		case "HEAD":
			e.HEAD(rule.Path, handler)
		case "OPTIONS":
			e.OPTIONS(rule.Path, handler)
		}
	}

	return nil
}

// createEchoHandler creates an Echo-specific handler for an endpoint
func (b *BaitRoute) createEchoHandler(rule Rule) echo.HandlerFunc {
	return func(c echo.Context) error {
		// Set custom headers
		for key, value := range rule.Headers {
			c.Response().Header().Set(key, value)
		}

		// Set content type
		if rule.ContentType != "" {
			c.Response().Header().Set("Content-Type", rule.ContentType)
		}

		// Trigger alert if handler is set
		if b.alertHandler != nil {
			alert := Alert{
				Method:        c.Request().Method,
				Path:          c.Request().URL.Path,
				SourceIP:      c.RealIP(),
				TrueClientIP:  c.Request().Header.Get("True-Client-IP"),
				XForwardedFor: c.Request().Header.Get("X-Forwarded-For"),
				RuleName:      b.ruleSources[c.Request().Method+":"+c.Request().URL.Path],
				Timestamp:     time.Now(),
			}
			go b.alertHandler(alert)
		}

		// Send response with status code and body
		return c.String(rule.Status, rule.Body)
	}
}
