package baitroute

import (
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// BaitRoute represents a bait route instance
type BaitRoute struct {
	rules        []Rule
	ruleSources  map[string]string
	alertHandler AlertHandler
	alerts       []Alert
}

// NewBaitRoute creates a new BaitRoute instance with rules loaded from the specified directory
func NewBaitRoute(rulesPath string, selectedRules ...string) (*BaitRoute, error) {
	b := &BaitRoute{
		rules:       make([]Rule, 0),
		ruleSources: make(map[string]string),
	}

	useAllRules := len(selectedRules) == 0
	err := filepath.Walk(rulesPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() || !strings.HasSuffix(strings.ToLower(info.Name()), ".yaml") {
			return nil
		}

		relPath := strings.TrimSuffix(strings.TrimPrefix(path, rulesPath+string(os.PathSeparator)), ".yaml")
		if !useAllRules && !contains(selectedRules, relPath) {
			return nil
		}

		content, err := ioutil.ReadFile(path)
		if err != nil {
			return fmt.Errorf("failed to read rule file %s: %v", relPath, err)
		}

		var rules []Rule
		if err := yaml.Unmarshal(content, &rules); err != nil {
			return fmt.Errorf("failed to parse rule file %s: %v", relPath, err)
		}

		for _, rule := range rules {
			key := fmt.Sprintf("%s:%s", rule.Method, rule.Path)
			if existingSource, exists := b.ruleSources[key]; exists {
				return &EndpointConflictError{
					Method:     rule.Method,
					Path:       rule.Path,
					SourceFile: existingSource,
				}
			}
			b.ruleSources[key] = relPath
		}

		b.rules = append(b.rules, rules...)
		return nil
	})

	if err != nil {
		return nil, err
	}

	const banner = ` ....
''. :   __
   \|_.'  ':       _.----._//_          Utku Sen's
  .'  .'.'-._   .'  _/ -._ \)-.----O   ___       _ _   ___          _
 '._.'.'      '--''-'._   '--..--'    | _ ) __ _(_) |_| _ \___ _  _| |_ ___
  .'.'___    /'---'. / ,-'            | _ \/ _  | |  _|   / _ \ || |  _/ -_)
_<__.-._))../ /'----'/.'_____:'.      |___/\__,_|_|\__|_|_\___/\_,_|\__\___|
:            \ ]              :  '.     
:  Acme       \\              :    '.  A web honeypot library to create decoy
:              \\__           :    .'  endpoints to detect and mislead attackers
:_______________|__]__________:...'
`
	fmt.Println()
	fmt.Print(banner)
	fmt.Printf("Successfully loaded %d bait endpoints\n", len(b.rules))
	fmt.Println()
	return b, nil
}

// OnBaitHit sets the handler for bait endpoint hits
func (b *BaitRoute) OnBaitHit(handler AlertHandler) {
	b.alertHandler = handler
	// Send alerts for any existing hits
	for _, alert := range b.alerts {
		b.sendAlert(alert)
	}
}

// sendAlert sends an alert to the registered alert handler
func (b *BaitRoute) sendAlert(alert Alert) {
	if b.alertHandler != nil {
		b.alertHandler(alert)
	} else {
		b.alerts = append(b.alerts, alert)
	}
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// GetMatchingRule finds a matching rule for the given path and method
func (b *BaitRoute) GetMatchingRule(path string, method string) *Rule {
	for _, rule := range b.rules {
		if rule.Path == path && rule.Method == method {
			return &rule
		}
	}
	return nil
}

// Handler handles HTTP requests and checks if they match any bait endpoints
func (b *BaitRoute) Handler(w http.ResponseWriter, r *http.Request) {
	// Check if this is a bait endpoint
	rule := b.GetMatchingRule(r.URL.Path, r.Method)
	if rule != nil {
		// Create and send alert
		alert := Alert{
			Method:        r.Method,
			Path:          r.URL.Path,
			SourceIP:      r.RemoteAddr,
			TrueClientIP:  r.Header.Get("True-Client-IP"),
			XForwardedFor: r.Header.Get("X-Forwarded-For"),
			RuleName:      b.ruleSources[r.Method+":"+r.URL.Path],
			Timestamp:     time.Now(),
		}

		// Read body if present
		if r.Body != nil {
			body, err := io.ReadAll(r.Body)
			if err == nil {
				alert.Body = string(body)
			}
		}

		b.sendAlert(alert)

		// Set response headers
		for key, value := range rule.Headers {
			w.Header().Set(key, value)
		}

		// Set content type
		if rule.ContentType != "" {
			w.Header().Set("Content-Type", rule.ContentType)
		}

		// Write response
		w.WriteHeader(rule.Status)
		w.Write([]byte(rule.Body))
		return
	}

	// Not a bait endpoint, pass to next handler
	http.NotFound(w, r)
}
