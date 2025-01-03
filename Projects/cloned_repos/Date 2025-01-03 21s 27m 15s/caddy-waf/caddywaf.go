package caddywaf

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/caddyserver/caddy/v2"
	"github.com/caddyserver/caddy/v2/caddyconfig/caddyfile"
	"github.com/caddyserver/caddy/v2/caddyconfig/httpcaddyfile"
	"github.com/caddyserver/caddy/v2/modules/caddyhttp"
	"github.com/oschwald/maxminddb-golang"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
)

func init() {
	fmt.Println("Registering WAF Middleware")
	caddy.RegisterModule(Middleware{})
	httpcaddyfile.RegisterHandlerDirective("waf", parseCaddyfile)
	fmt.Println("WAF Middleware Registered Successfully")
}

var (
	_ caddy.Provisioner           = (*Middleware)(nil)
	_ caddyhttp.MiddlewareHandler = (*Middleware)(nil)
	_ caddyfile.Unmarshaler       = (*Middleware)(nil)
)

type RateLimit struct {
	Requests int           `json:"requests"`
	Window   time.Duration `json:"window"`
}

type requestCounter struct {
	count  int
	window time.Time
}

type RateLimiter struct {
	sync.RWMutex
	requests map[string]*requestCounter
	config   RateLimit
}

type CountryAccessFilter struct {
	Enabled     bool     `json:"enabled"`
	CountryList []string `json:"country_list"`
	GeoIPDBPath string   `json:"geoip_db_path"`
	geoIP       *maxminddb.Reader
}

type GeoIPRecord struct {
	Country struct {
		ISOCode string `maxminddb:"iso_code"`
	} `maxminddb:"country"`
}

type Rule struct {
	ID          string   `json:"id"`
	Phase       int      `json:"phase"`
	Pattern     string   `json:"pattern"`
	Targets     []string `json:"targets"`
	Severity    string   `json:"severity"`
	Action      string   `json:"action"`
	Score       int      `json:"score"`
	Mode        string   `json:"mode"`
	Description string   `json:"description"`
	regex       *regexp.Regexp
}

type SeverityConfig struct {
	Critical string `json:"critical,omitempty"`
	High     string `json:"high,omitempty"`
	Medium   string `json:"medium,omitempty"`
	Low      string `json:"low,omitempty"`
}

type Middleware struct {
	RuleFiles        []string            `json:"rule_files"`
	IPBlacklistFile  string              `json:"ip_blacklist_file"`
	DNSBlacklistFile string              `json:"dns_blacklist_file"`
	LogAll           bool                `json:"log_all"`
	AnomalyThreshold int                 `json:"anomaly_threshold"`
	RateLimit        RateLimit           `json:"rate_limit"`
	CountryBlock     CountryAccessFilter `json:"country_block"`
	CountryWhitelist CountryAccessFilter `json:"country_whitelist"`
	Severity         SeverityConfig      `json:"severity,omitempty"`
	Rules            []Rule              `json:"-"`
	ipBlacklist      map[string]bool     `json:"-"`
	dnsBlacklist     []string            `json:"-"`
	rateLimiter      *RateLimiter        `json:"-"`
	logger           *zap.Logger
}

func (Middleware) CaddyModule() caddy.ModuleInfo {
	return caddy.ModuleInfo{
		ID:  "http.handlers.waf",
		New: func() caddy.Module { return &Middleware{} },
	}
}

func parseCaddyfile(h httpcaddyfile.Helper) (caddyhttp.MiddlewareHandler, error) {
	var m Middleware
	err := m.UnmarshalCaddyfile(h.Dispenser)
	if err != nil {
		return nil, err
	}
	return &m, nil
}

func (m *Middleware) UnmarshalCaddyfile(d *caddyfile.Dispenser) error {
	fmt.Println("WAF UnmarshalCaddyfile Called")
	for d.Next() {
		for d.NextBlock(0) {
			switch d.Val() {
			case "rate_limit":
				if !d.NextArg() {
					return d.ArgErr()
				}
				requests, err := strconv.Atoi(d.Val())
				if err != nil {
					return d.Errf("invalid rate limit request count: %v", err)
				}
				if !d.NextArg() {
					return d.ArgErr()
				}
				window, err := time.ParseDuration(d.Val())
				if err != nil {
					return d.Errf("invalid rate limit window: %v", err)
				}
				m.RateLimit = RateLimit{
					Requests: requests,
					Window:   window,
				}
			case "block_countries":
				m.CountryBlock.Enabled = true
				if !d.NextArg() {
					return d.ArgErr()
				}
				m.CountryBlock.GeoIPDBPath = d.Val()
				for d.NextArg() {
					m.CountryBlock.CountryList = append(m.CountryBlock.CountryList, strings.ToUpper(d.Val()))
				}
			case "whitelist_countries":
				m.CountryWhitelist.Enabled = true
				if !d.NextArg() {
					return d.ArgErr()
				}
				m.CountryWhitelist.GeoIPDBPath = d.Val()
				for d.NextArg() {
					m.CountryWhitelist.CountryList = append(m.CountryWhitelist.CountryList, strings.ToUpper(d.Val()))
				}
			case "log_all":
				fmt.Println("WAF Log All Enabled")
				m.LogAll = true
			case "rule_file":
				fmt.Println("WAF Loading Rule File")
				if !d.NextArg() {
					return d.ArgErr()
				}
				m.RuleFiles = append(m.RuleFiles, d.Val())
			case "ip_blacklist_file":
				fmt.Println("WAF Loading IP Blacklist File")
				if !d.NextArg() {
					return d.ArgErr()
				}
				m.IPBlacklistFile = d.Val()
			case "dns_blacklist_file":
				fmt.Println("WAF Loading DNS Blacklist File")
				if !d.NextArg() {
					return d.ArgErr()
				}
				m.DNSBlacklistFile = d.Val()
			case "severity":
				if !d.NextArg() {
					return d.ArgErr()
				}
				severityLevel := strings.ToLower(d.Val())
				if !d.NextArg() {
					return d.ArgErr()
				}
				action := strings.ToLower(d.Val())
				switch severityLevel {
				case "critical":
					m.Severity.Critical = action
				case "high":
					m.Severity.High = action
				case "medium":
					m.Severity.Medium = action
				case "low":
					m.Severity.Low = action
				default:
					return d.Errf("invalid severity level: %s", severityLevel)
				}
			default:
				fmt.Println("WAF Unrecognized SubDirective: ", d.Val())
				return d.Errf("unrecognized subdirective: %s", d.Val())
			}
		}
	}
	return nil
}

func (rl *RateLimiter) isRateLimited(ip string) bool {
	rl.Lock()
	defer rl.Unlock()

	now := time.Now()
	if counter, exists := rl.requests[ip]; exists {
		if now.Sub(counter.window) > rl.config.Window {
			counter.count = 1
			counter.window = now
			return false
		}
		counter.count++
		return counter.count > rl.config.Requests
	}

	rl.requests[ip] = &requestCounter{
		count:  1,
		window: now,
	}
	return false
}

func (m *Middleware) isCountryInList(remoteAddr string, countryList []string, geoIP *maxminddb.Reader) (bool, error) {
	if geoIP == nil {
		return false, fmt.Errorf("GeoIP database not loaded")
	}

	ip, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		ip = remoteAddr
	}

	parsedIP := net.ParseIP(ip)
	if parsedIP == nil {
		return false, fmt.Errorf("invalid IP address: %s", ip)
	}

	var record GeoIPRecord
	err = geoIP.Lookup(parsedIP, &record)
	if err != nil {
		return false, err
	}

	for _, country := range countryList {
		if strings.EqualFold(record.Country.ISOCode, country) {
			return true, nil
		}
	}

	return false, nil
}

func (m *Middleware) ServeHTTP(w http.ResponseWriter, r *http.Request, next caddyhttp.Handler) error {
	if m.handlePhase1(w, r) {
		return nil
	}

	totalScore := m.handlePhase2(w, r)
	if totalScore >= m.AnomalyThreshold {
		m.handlePhase3(w, r)
		return nil
	}

	return next.ServeHTTP(w, r)
}

func (m *Middleware) handlePhase1(w http.ResponseWriter, r *http.Request) bool {
	if m.CountryBlock.Enabled {
		blocked, err := m.isCountryInList(r.RemoteAddr, m.CountryBlock.CountryList, m.CountryBlock.geoIP)

		if err != nil {
			m.logRequest(zapcore.ErrorLevel, "Failed to check country block",
				zap.String("ip", r.RemoteAddr),
				zap.Error(err),
			)
		} else if blocked {
			m.logRequest(zapcore.InfoLevel, "Request blocked by country",
				zap.String("ip", r.RemoteAddr),
				zap.Int("status_code", http.StatusForbidden),
			)
			w.WriteHeader(http.StatusForbidden)
			return true
		}
	}

	if m.CountryWhitelist.Enabled {
		whitelisted, err := m.isCountryInList(r.RemoteAddr, m.CountryWhitelist.CountryList, m.CountryWhitelist.geoIP)

		if err != nil {
			m.logRequest(zapcore.ErrorLevel, "Failed to check country whitelist",
				zap.String("ip", r.RemoteAddr),
				zap.Error(err),
			)
		} else if !whitelisted {
			m.logRequest(zapcore.InfoLevel, "Request blocked by country whitelist",
				zap.String("ip", r.RemoteAddr),
				zap.Int("status_code", http.StatusForbidden),
			)
			w.WriteHeader(http.StatusForbidden)
			return true
		}
	}

	if m.rateLimiter != nil {
		ip, _, err := net.SplitHostPort(r.RemoteAddr)
		if err == nil && m.rateLimiter.isRateLimited(ip) {
			m.logRequest(zapcore.InfoLevel, "Request blocked by rate limit",
				zap.String("ip", ip),
				zap.Int("status_code", http.StatusTooManyRequests),
			)
			w.WriteHeader(http.StatusTooManyRequests)
			return true
		}
	}

	if m.isIPBlacklisted(r.RemoteAddr) {
		m.logRequest(zapcore.InfoLevel, "Request blocked by IP blacklist",
			zap.String("ip", r.RemoteAddr),
			zap.Int("status_code", http.StatusForbidden),
		)
		w.WriteHeader(http.StatusForbidden)
		return true
	}

	if m.isDNSBlacklisted(r.Host) {
		m.logRequest(zapcore.InfoLevel, "Request blocked by DNS blacklist",
			zap.String("domain", r.Host),
			zap.Int("status_code", http.StatusForbidden),
		)
		w.WriteHeader(http.StatusForbidden)
		return true
	}

	return false
}

func (m *Middleware) handlePhase2(w http.ResponseWriter, r *http.Request) int {
	totalScore := 0
	for _, rule := range m.Rules {
		if rule.Phase != 2 {
			continue
		}
		for _, target := range rule.Targets {
			value, _ := m.extractValue(target, r)
			if rule.regex.MatchString(value) {
				totalScore += rule.Score
				action := rule.Action
				if action == "" {
					action = m.getSeverityAction(rule.Severity)
				}
				switch action {
				case "block":
					m.logRequest(zapcore.InfoLevel, "Rule Matched",
						zap.String("rule_id", rule.ID),
						zap.String("target", target),
						zap.String("value", value),
						zap.String("description", rule.Description),
						zap.Int("status_code", http.StatusForbidden),
					)
					w.WriteHeader(http.StatusForbidden)
					return totalScore
				case "log":
					m.logRequest(zapcore.InfoLevel, "Rule Matched",
						zap.String("rule_id", rule.ID),
						zap.String("target", target),
						zap.String("value", value),
						zap.String("description", rule.Description),
						zap.Int("status_code", http.StatusOK),
					)
				}
			}
		}
	}
	return totalScore
}

func (m *Middleware) handlePhase3(w http.ResponseWriter, r *http.Request) {
	m.logRequest(zapcore.InfoLevel, "Request blocked by Anomaly Threshold",
		zap.Int("status_code", http.StatusForbidden),
	)
	w.WriteHeader(http.StatusForbidden)
}

func (m *Middleware) logRequest(level zapcore.Level, msg string, fields ...zap.Field) {
	switch level {
	case zapcore.DebugLevel:
		m.logger.Debug(msg, fields...)
	case zapcore.InfoLevel:
		m.logger.Info(msg, fields...)
	case zapcore.WarnLevel:
		m.logger.Warn(msg, fields...)
	case zapcore.ErrorLevel:
		m.logger.Error(msg, fields...)
	default:
		m.logger.Info(msg, fields...)
	}
}

func (m *Middleware) getSeverityAction(severity string) string {
	switch strings.ToLower(severity) {
	case "critical":
		return m.Severity.Critical
	case "high":
		return m.Severity.High
	case "medium":
		return m.Severity.Medium
	case "low":
		return m.Severity.Low
	default:
		return "log"
	}
}

func (m *Middleware) Provision(ctx caddy.Context) error {
	m.logger = ctx.Logger()

	if m.RateLimit.Requests > 0 {
		m.rateLimiter = &RateLimiter{
			requests: make(map[string]*requestCounter),
			config:   m.RateLimit,
		}
	}

	if m.CountryBlock.Enabled {
		reader, err := maxminddb.Open(m.CountryBlock.GeoIPDBPath)
		if err != nil {
			return fmt.Errorf("failed to load GeoIP database: %v", err)
		}
		m.CountryBlock.geoIP = reader
	}

	for _, file := range m.RuleFiles {
		if err := m.loadRulesFromFile(file); err != nil {
			return fmt.Errorf("failed to load rules from %s: %v", file, err)
		}
	}

	if m.AnomalyThreshold == 0 {
		m.AnomalyThreshold = 5
	}

	if m.IPBlacklistFile != "" {
		if err := m.loadIPBlacklistFromFile(m.IPBlacklistFile); err != nil {
			return fmt.Errorf("failed to load IP blacklist from %s: %v", m.IPBlacklistFile, err)
		}
	} else {
		m.ipBlacklist = make(map[string]bool)
	}

	if m.DNSBlacklistFile != "" {
		if err := m.loadDNSBlacklistFromFile(m.DNSBlacklistFile); err != nil {
			return fmt.Errorf("failed to load DNS blacklist from %s: %v", m.DNSBlacklistFile, err)
		}
	} else {
		m.dnsBlacklist = []string{}
	}

	return nil
}

func (m *Middleware) isIPBlacklisted(remoteAddr string) bool {
	if len(m.ipBlacklist) == 0 {
		return false
	}
	ip, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		return false
	}
	if m.ipBlacklist[ip] {
		return true
	}
	parsedIP := net.ParseIP(ip)
	if parsedIP == nil {
		return false
	}

	for blacklistIP := range m.ipBlacklist {
		_, ipNet, err := net.ParseCIDR(blacklistIP)
		if err != nil {
			if blacklistIP == ip {
				return true
			}
			continue
		}
		if ipNet.Contains(parsedIP) {
			return true
		}
	}
	return false
}

func (m *Middleware) isDNSBlacklisted(host string) bool {
	if m.dnsBlacklist == nil || len(m.dnsBlacklist) == 0 {
		return false
	}
	for _, blacklistedDomain := range m.dnsBlacklist {
		if strings.EqualFold(host, blacklistedDomain) {
			return true
		}
	}
	return false
}

func (m *Middleware) extractValue(target string, r *http.Request) (string, error) {
	switch target {
	case "ARGS":
		return r.URL.RawQuery, nil
	case "BODY":
		if r.Body == nil {
			return "", nil
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return "", err
		}
		r.Body = io.NopCloser(bytes.NewReader(body))
		return string(body), nil
	case "HEADERS":
		return fmt.Sprintf("%v", r.Header), nil
	case "URL":
		return r.URL.Path, nil
	case "USER_AGENT":
		return r.UserAgent(), nil
	case "COOKIES":
		return fmt.Sprintf("%v", r.Cookies()), nil
	case "PATH":
		return r.URL.Path, nil
	case "URI":
		return r.RequestURI, nil
	default:
		return "", fmt.Errorf("unknown target: %s", target)
	}
}

func (m *Middleware) loadRulesFromFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var rules []Rule
	if err := json.Unmarshal(content, &rules); err != nil {
		return err
	}
	for i, rule := range rules {
		regex, err := regexp.Compile(rule.Pattern)
		if err != nil {
			return fmt.Errorf("invalid pattern in rule %s: %v", rule.ID, err)
		}
		rules[i].regex = regex
		if rule.Mode == "" {
			rules[i].Mode = rule.Action
		}
	}
	m.Rules = append(m.Rules, rules...)
	return nil
}

func (m *Middleware) loadIPBlacklistFromFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	ips := strings.Split(string(content), "\n")
	m.ipBlacklist = make(map[string]bool)
	for _, ip := range ips {
		if ip != "" {
			m.ipBlacklist[ip] = true
		}
	}
	return nil
}

func (m *Middleware) loadDNSBlacklistFromFile(path string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	m.dnsBlacklist = strings.Split(string(content), "\n")
	return nil
}

func (m *Middleware) ReloadConfig() error {
	if err := m.loadRulesFromFiles(); err != nil {
		return err
	}
	if err := m.loadIPBlacklistFromFile(m.IPBlacklistFile); err != nil {
		return err
	}
	if err := m.loadDNSBlacklistFromFile(m.DNSBlacklistFile); err != nil {
		return err
	}
	return nil
}

func (m *Middleware) loadRulesFromFiles() error {
	m.Rules = []Rule{}
	for _, file := range m.RuleFiles {
		if err := m.loadRulesFromFile(file); err != nil {
			return err
		}
	}
	return nil
}
