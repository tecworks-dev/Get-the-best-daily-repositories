package baitroute

import (
	"time"

	"github.com/gin-gonic/gin"
)

// RegisterWithGin registers bait endpoints with a Gin router
func (b *BaitRoute) RegisterWithGin(router *gin.Engine) error {
	// Register bait endpoints
	for _, rule := range b.rules {
		handler := b.createGinHandler(rule)

		switch rule.Method {
		case "GET":
			router.GET(rule.Path, handler)
		case "POST":
			router.POST(rule.Path, handler)
		case "PUT":
			router.PUT(rule.Path, handler)
		case "DELETE":
			router.DELETE(rule.Path, handler)
		case "PATCH":
			router.PATCH(rule.Path, handler)
		case "HEAD":
			router.HEAD(rule.Path, handler)
		case "OPTIONS":
			router.OPTIONS(rule.Path, handler)
		}
	}

	return nil
}

// createGinHandler creates a Gin-specific handler for an endpoint
func (b *BaitRoute) createGinHandler(rule Rule) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Set custom headers
		for key, value := range rule.Headers {
			c.Header(key, value)
		}

		// Set content type
		if rule.ContentType != "" {
			c.Header("Content-Type", rule.ContentType)
		}

		// Trigger alert if handler is set
		if b.alertHandler != nil {
			alert := Alert{
				Method:        c.Request.Method,
				Path:          c.Request.URL.Path,
				SourceIP:      c.ClientIP(),
				TrueClientIP:  c.GetHeader("True-Client-IP"),
				XForwardedFor: c.GetHeader("X-Forwarded-For"),
				RuleName:      b.ruleSources[c.Request.Method+":"+c.Request.URL.Path],
				Timestamp:     time.Now(),
			}
			go b.alertHandler(alert)
		}

		// Send response with status code and body
		c.String(rule.Status, rule.Body)
	}
}
