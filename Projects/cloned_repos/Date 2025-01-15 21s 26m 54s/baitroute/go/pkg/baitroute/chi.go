package baitroute

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
)

// RegisterWithChi registers bait route endpoints with a Chi router
func (b *BaitRoute) RegisterWithChi(router chi.Router) error {
	// Register bait endpoints
	for _, rule := range b.rules {
		handler := b.createChiHandler(rule)

		switch rule.Method {
		case "GET":
			router.Get(rule.Path, handler)
		case "POST":
			router.Post(rule.Path, handler)
		case "PUT":
			router.Put(rule.Path, handler)
		case "DELETE":
			router.Delete(rule.Path, handler)
		case "PATCH":
			router.Patch(rule.Path, handler)
		case "HEAD":
			router.Head(rule.Path, handler)
		case "OPTIONS":
			router.Options(rule.Path, handler)
		}
	}

	return nil
}

// createChiHandler creates a Chi-specific handler for an endpoint
func (b *BaitRoute) createChiHandler(rule Rule) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Set custom headers
		for key, value := range rule.Headers {
			w.Header().Set(key, value)
		}

		// Set content type
		if rule.ContentType != "" {
			w.Header().Set("Content-Type", rule.ContentType)
		}

		// Trigger alert if handler is set
		if b.alertHandler != nil {
			alert := Alert{
				Method:        r.Method,
				Path:          r.URL.Path,
				SourceIP:      r.RemoteAddr,
				TrueClientIP:  r.Header.Get("True-Client-IP"),
				XForwardedFor: r.Header.Get("X-Forwarded-For"),
				RuleName:      b.ruleSources[r.Method+":"+r.URL.Path],
				Timestamp:     time.Now(),
			}
			go b.alertHandler(alert)
		}

		// Send response with status code and body
		w.WriteHeader(rule.Status)
		w.Write([]byte(rule.Body))
	}
}
