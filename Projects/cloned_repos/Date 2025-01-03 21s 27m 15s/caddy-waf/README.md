# 🛡️ Caddy WAF Middleware

A **simple Web Application Firewall (WAF)** middleware for the Caddy server, designed to provide **comprehensive protection** against web attacks. This middleware integrates seamlessly with Caddy and offers a wide range of security features to safeguard your applications.

---

## 🌟 Key Features

- **Rule-based request filtering** with regex patterns.
- **IP and DNS blacklisting** to block malicious traffic.
- **Country-based blocking** using MaxMind GeoIP2.
- **Rate limiting** per IP address to prevent abuse.
- **Anomaly scoring system** for detecting suspicious behavior.
- **Request inspection** (URL, args, body, headers, cookies, user-agent).
- **Protection against common attacks** (SQL injection, XSS, RCE, Log4j, etc.).
- **Detailed logging and monitoring** for security analysis.
- **Dynamic rule reloading** without server restart.
- **Severity-based actions** (block, log) for fine-grained control.

## 🚀 Installation

```bash
# Step 1: Clone the caddy-waf repository from GitHub
# This downloads the source code for the caddy-waf module to your local machine.
git clone https://github.com/fabriziosalmi/caddy-waf.git

# Step 2: Navigate into the caddy-waf directory
# This changes your current working directory to the caddy-waf project folder.
cd caddy-waf

# Step 3: Clean up and update the go.mod file
# This ensures that your project's dependencies are up-to-date and consistent.
go mod tidy

# Step 4: Fetch and install the required Go modules
# This downloads and installs the caddy-waf module, Caddy v2, and the maxminddb-golang library.
go get -v github.com/fabriziosalmi/caddy-waf github.com/caddyserver/caddy/v2 github.com/oschwald/maxminddb-golang

# Step 5: Download the GeoLite2 Country database
# This downloads the GeoLite2 Country database, which is required for IP-based country filtering.
wget https://git.io/GeoLite2-Country.mmdb

# Step 6: Clean up previous build artifacts
# This removes any temporary build directories from previous builds to ensure a clean build environment.
rm -rf buildenv_*

# Step 7: Build Caddy with the caddy-waf module
# This compiles Caddy and includes the caddy-waf module as a custom plugin.
xcaddy build --with github.com/fabriziosalmi/caddy-waf

# Step 8: Run the compiled Caddy server
# This starts the Caddy server with the caddy-waf module enabled.
./caddy run
```

### Final Notes

- If you encounter any issues, ensure that your Go environment is set up correctly and that you're using a compatible version of Go (as specified in the `caddy-waf` repository's `go.mod` file).
- After building Caddy with `xcaddy`, the resulting binary will include the WAF middleware. You can verify this by running:
  ```bash
  ./caddy list-modules
  ```
  Look for the `http.handlers.waf` module in the output.


## 🛠️ Configuration

### Basic Caddyfile Setup

```caddyfile
{
    auto_https off
    admin off
    debug
}

:8080 {
    log {
        output stdout
        format console
        level DEBUG
    }
    
    route {
        waf {
            # Rate limiting: 100 requests per 5 seconds
            rate_limit 100 5s
            
            # Rules and blacklists
            rule_file rules.json
            ip_blacklist_file ip_blacklist.txt
            dns_blacklist_file dns_blacklist.txt
            
            # Country blocking (requires MaxMind GeoIP2 database)
            block_countries GeoLite2-Country.mmdb RU CN NK

            # Whitelist countries (requires MaxMind GeoIP2 database)
            # whitelist_countries GeoLite2-Country.mmdb US
            
            # Enable detailed logging
            log_all

            # Define actions based on severity
            severity critical block
            severity high block
            severity medium log
            severity low log
        }
        respond "Hello, world!" 200
    }
}
```

### Reverse Proxy with WAF Setup

```caddyfile
# Global options
{
    # Enable the global error log
    log {
        output file /var/log/caddy/errors.log
        level ERROR
    }
    # Automatic HTTPS settings
    email admin@example.com
}

# Reverse proxy for example.com
example.com {
    # Enable WAF
    waf {
        # Rate limiting: 100 requests per 5 seconds
        rate_limit 100 5s

        # Rules and blacklists
        rule_file /path/to/rules.json
        ip_blacklist_file /path/to/ip_blacklist.txt
        dns_blacklist_file /path/to/dns_blacklist.txt

        # Country blocking (requires MaxMind GeoIP2 database)
        block_countries /path/to/GeoLite2-Country.mmdb RU CN NK

        # Enable detailed logging
        log_all

        # Define actions based on severity
        severity critical block
        severity high block
        severity medium log
        severity low log
    }

    # Log access to a file
    log {
        output file /var/log/caddy/access.log
        format single_field common_log
    }

    # Reverse proxy to the origin server
    reverse_proxy http://origin-server:8080 {
        # Optional: Add headers to forward to the backend
        header_up X-Real-IP {remote_host}
        header_up X-Forwarded-For {remote_host}
        header_up X-Forwarded-Proto {scheme}
    }
}
```

## ⚙️ Configuration Options

| Option | Description | Example |
|--------|-------------|---------|
| `rule_file` | JSON file containing WAF rules | `rule_file rules.json` |
| `ip_blacklist_file` | File with blocked IPs/CIDR ranges | `ip_blacklist_file blacklist.txt` |
| `dns_blacklist_file` | File with blocked domains | `dns_blacklist_file domains.txt` |
| `rate_limit` | Rate limiting config | `rate_limit 100 1m` |
| `block_countries` | Country blocking config | `block_countries GeoLite2-Country.mmdb RU CN NK` |
| `log_all` | Enable detailed logging | `log_all` |
| `severity` | Define actions based on severity levels | `severity critical block` |


## 📜 Rules Format (`rules.json`)

Rules are defined in a JSON file. Each rule specifies a pattern to match, targets to inspect, and actions to take.

```json
[
    {
        "id": "sql_injection",
        "phase": 1,
        "pattern": "(?i)(?:select|insert|update|delete|drop|alter)(?:[\\s\\v\\/\\*]+)(?:from|into|where|table)\\b",
        "targets": ["ARGS", "BODY", "HEADERS", "COOKIES"],
        "severity": "CRITICAL",
        "action": "block",
        "score": 10,
        "description": "Block SQL injection attempts."
    }
]
```

### Rule Fields

| Field | Description | Example |
|-------|-------------|---------|
| `id` | Unique rule identifier | `sql_injection` |
| `phase` | Processing phase (1-5) | `1` |
| `pattern` | Regular expression pattern | `(?i)(?:select|insert)` |
| `targets` | Areas to inspect | `["ARGS", "BODY"]` |
| `severity` | Rule severity (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`) | `CRITICAL` |
| `action` | Action to take (`block`, `log`) | `block` |
| `score` | Score for anomaly detection | `10` |
| `description` | Rule description | `Block SQL injection attempts.` |


## 🛡️ Protected Attack Types

1. **SQL Injection**
   - Basic SELECT/UNION injections
   - Time-based injection attacks
   - Boolean-based injections

2. **Cross-Site Scripting (XSS)**
   - Script tag injection
   - Event handler injection
   - SVG-based XSS

3. **Path Traversal**
   - Directory traversal attempts
   - Encoded path traversal
   - Double-encoded traversal

4. **Remote Code Execution (RCE)**
   - Command injection
   - Shell command execution
   - System command execution

5. **Log4j Exploits**
   - JNDI lookup attempts
   - Nested expressions

6. **Protocol Attacks**
   - Git repository access
   - Environment file access
   - Configuration file access

7. **Scanner Detection**
   - Common vulnerability scanners
   - Web application scanners
   - Network scanning tools


## 🚫 Blacklist Formats

### IP Blacklist (`ip_blacklist.txt`)
```text
192.168.1.1
10.0.0.0/8
2001:db8::/32
```

### DNS Blacklist (`dns_blacklist.txt`)
```text
malicious.com
evil.example.org
```

## ⏱️ Rate Limiting

Configure rate limits using requests count and time window:

```caddyfile
# 100 requests per minute
rate_limit 100 1m

# 10 requests per second
rate_limit 10 1s

# 1000 requests per hour
rate_limit 1000 1h
```

## 🌍 Country Blocking

Block traffic from specific countries using ISO country codes:

```caddyfile
# Block requests from Russia, China, and North Korea
block_countries /path/to/GeoLite2-Country.mmdb RU CN KP
```

## 🔄 Dynamic Updates

Rules and blacklists can be updated without server restart:
1. Modify `rules.json` or blacklist files.
2. Reload Caddy: `caddy reload`.

## 🧪 Testing

### Basic Testing
```bash
# Test rate limiting
for i in {1..10}; do curl -i http://localhost:8080/; done

# Test country blocking
curl -H "X-Real-IP: 1.2.3.4" http://localhost:8080/

# Test SQL injection protection
curl "http://localhost:8080/?id=1+UNION+SELECT+*+FROM+users"

# Test XSS protection
curl "http://localhost:8080/?input=<script>alert(1)</script>"
```

### Load Testing
```bash
ab -n 1000 -c 100 http://localhost:8080/
```

## 📜 License

This project is licensed under the **AGPLv3 License**.

## 🙏 Contributing

Contributions are welcome! Please open an issue or submit a pull request.

