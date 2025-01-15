package baitroute

import (
	"time"

	"github.com/valyala/fasthttp"
)

// RegisterWithFastHTTP registers bait route endpoints with FastHTTP request handler
func (b *BaitRoute) RegisterWithFastHTTP() fasthttp.RequestHandler {
	return func(ctx *fasthttp.RequestCtx) {
		path := string(ctx.Path())
		method := string(ctx.Method())

		// Find matching endpoint
		for _, rule := range b.rules {
			if rule.Path == path && rule.Method == method {
				// Set custom headers
				for key, value := range rule.Headers {
					ctx.Response.Header.Set(key, value)
				}

				// Set content type
				if rule.ContentType != "" {
					ctx.Response.Header.Set("Content-Type", rule.ContentType)
				}

				// Set status code
				ctx.Response.SetStatusCode(rule.Status)

				// Trigger alert if handler is set
				if b.alertHandler != nil {
					// Convert headers to map
					headers := make(map[string][]string)
					ctx.Request.Header.VisitAll(func(key, value []byte) {
						headers[string(key)] = append(headers[string(key)], string(value))
					})

					alert := Alert{
						Method:        method,
						Path:          path,
						SourceIP:      ctx.RemoteIP().String(),
						TrueClientIP:  string(ctx.Request.Header.Peek("True-Client-IP")),
						XForwardedFor: string(ctx.Request.Header.Peek("X-Forwarded-For")),
						RuleName:      b.ruleSources[method+":"+path],
						Timestamp:     time.Now(),
					}
					go b.alertHandler(alert)
				}

				// Write response body
				ctx.WriteString(rule.Body)
				return
			}
		}

		// No matching endpoint found, pass to next handler
		ctx.Response.SetStatusCode(fasthttp.StatusNotFound)
	}
}
