package baitroute

import (
	"net/http"
	"time"
)

// RegisterWithHTTP registers bait endpoints with a standard http.ServeMux
func (b *BaitRoute) RegisterWithHTTP(mux *http.ServeMux) error {
	for _, rule := range b.rules {
		handler := createHTTPHandler(b, rule)
		mux.Handle(rule.Path, handler)
	}
	return nil
}

// createHTTPHandler creates a http.HandlerFunc for the given endpoint
func createHTTPHandler(b *BaitRoute, rule Rule) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Check if the method matches
		if r.Method != rule.Method {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}

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
